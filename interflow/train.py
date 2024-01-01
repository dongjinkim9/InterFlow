import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataset import RealMultiscaleSRDataset, crop, augmentation
from models.flow_model import InterpFlow
import numpy as np
from PIL import Image
import random
import wandb
from utils.resize_right import resize
from utils.interp_methods import cubic
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(args):
    # set logger
    wandb.init(project="Interpolating-Flow",
               config=args,
               name=args.parameter_name,
               notes=args.parameter_name,
               mode="disabled" if not args.wandb else None)
    artifact = wandb.Artifact(f'{args.parameter_name}_model', type='model',)

    writer = SummaryWriter('log/train_interflow')

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[*] Using GPU: {}'.format(args.gpu), flush=True)

    # set model
    model = InterpFlow(scales=args.scales,mean_init=args.mean_init, std_init=args.std_init)
    model = model.to(device)
    
    # load parameters
    if(not args.parameter_restore_path is None):
        model.load_state_dict(torch.load(args.parameter_restore_path))
        print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))
    if(not args.parameter_save_path is None and not os.path.exists(args.parameter_save_path)):
        os.makedirs(args.parameter_save_path)
    if(not args.rrdb_restore_path is None):
        model.rrdb.load_state_dict(torch.load(args.rrdb_restore_path))
    model.train()

    # set dataset
    transform = [crop(args.patch_size), augmentation()]
    dataset = RealMultiscaleSRDataset(args.GT_path, model.classes, preload=False, transform=transform, 
        dataset_size=args.train_iteration*args.batch_size, batch_size=args.batch_size)
    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # set optimizer
    train_parameters = list(filter(lambda x: x.requires_grad, list(model.parameters())))
    optimizer = optim.Adam(train_parameters, lr=args.learning_rate)
    learning_rate_scheduler = MultiStepLR(optimizer, milestones=[50000, 75000, 90000], gamma=0.5)

    loss_L1 = nn.L1Loss()

    print('\n[*] args: {}\n'.format(args))

    nan_count, skip_count = 0, 0
    save_path = None

    with tqdm(enumerate(loader)) as pbar:
        for step, train_data in pbar:
            gt = train_data['GT'].to(device)
            low_res = train_data['LR'].to(device)
            degradation_scale = train_data['scale'].to(device)

            lr_enc = model.arch.rrdbPreprocessing(gt)
            z, nll, logdet, ll_ib = model(low_res, gt, degradation_scale, lr_enc=lr_enc)
            ib_loss = -torch.mean(ll_ib)
            nll_loss = torch.mean(nll)

            # lr-consistency loss
            if args.consistency_lambda > 0:
                total_scales = sorted(model.classes)
                interp_scales = sorted(random.sample(total_scales,k=2))
                target_scale = torch.distributions.Uniform(*interp_scales).sample([gt.shape[0]]).to(device)

                img_lr1 = train_data[f'LR{interp_scales[0]}'].to(device)
                img_lr2 = train_data[f'LR{interp_scales[1]}'].to(device)
                z_lr1, _ = model(img_lr1, gt, torch.tensor(interp_scales[0:1] * args.batch_size), lr_enc=lr_enc, calc_loss = False)
                z_lr2, _ = model(img_lr2, gt, torch.tensor(interp_scales[1:2] * args.batch_size), lr_enc=lr_enc, calc_loss = False)
                
                interpolate_weight = (target_scale - interp_scales[0]) / (interp_scales[1] - interp_scales[0])
                weight_lr1 = (1-interpolate_weight).reshape(-1,1,1,1)
                weight_lr2 = interpolate_weight.reshape(-1,1,1,1)

                interpolated_z = weight_lr1*z_lr1 + weight_lr2*z_lr2

                interpolated_img, _ = model(interpolated_z, gt, reverse=True, test_input_size=low_res.shape[2], lr_enc=lr_enc)

                img_lr1 = train_data[f'LR{interp_scales[0]}'].to(device)
                img_lr2 = train_data[f'LR{interp_scales[1]}'].to(device)
                
                synth_lr = torch.zeros_like(low_res, device=device)
                synth_lr = synth_lr + weight_lr1*img_lr1 + weight_lr2*img_lr2
                synth_lr = synth_lr + ((torch.rand(synth_lr.shape, device=synth_lr.device) - 0.5) / 256)
                interp_down = resize(input=interpolated_img,scale_factors=1./4,interp_method=cubic,antialiasing=True)
                synth_down = resize(input=synth_lr,scale_factors=1./4,interp_method=cubic,antialiasing=True)
                consistency_loss = loss_L1(synth_down,interp_down)
                
                consistency_loss = consistency_loss.mean()
            else:
                consistency_loss = torch.tensor(0.)

            loss = nll_loss + args.consistency_lambda * consistency_loss + args.ib_lambda*ib_loss

            # block abnormal losses
            if not torch.isfinite(loss):
                nan_count += 1
                wandb.log({'errors/NaN count': nan_count, 'iter': step,})
                writer.add_scalar('NaN count', nan_count, step)
                if(nan_count > 1000):
                    raise Exception('More than 1000 nan count!')
                loss.backward()
                continue
            if step > 100 and loss > 30:
                skip_count += 1
                wandb.log({'errors/skip count': skip_count, 'iter': step,})
                writer.add_scalar('skip count', skip_count, step)
                loss.backward()
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()

            pbar.set_postfix(nll=nll_loss.item(), consistency_loss=consistency_loss.item(),
                             ib_loss=ib_loss.item() ,lr=optimizer.param_groups[0]['lr'])
            
            if (step+1) % 500 == 0:                  
                wandb.log({'train/nll': nll_loss.item(),
                        'train/consistency_loss': consistency_loss.item(),
                        'train/ib_loss': ib_loss.item(),
                        'train/total_loss': loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'iter': step,})
                
                writer.add_scalar('nll', nll_loss.item(), step)
                writer.add_scalar('consistency_loss', consistency_loss.item(), step)
                writer.add_scalar('ib_loss', ib_loss.item(), step)
                writer.add_scalar('total_loss', loss.item(), step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)

            if (step+1) % args.checkpoint_step == 0 and args.parameter_save_path is not None:
                save_path = os.path.join(args.parameter_save_path, 'last_{}'.format(args.parameter_name))
                torch.save(model.state_dict(), save_path)
    
    if save_path:
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[*] Using GPU: {}'.format(args.gpu), flush=True)

    model = InterpFlow(scales=args.scales)
    model = model.to(device)
    model.load_state_dict(torch.load(args.parameter_restore_path))
    print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))

    for scale in np.arange(2.1, 4.0, 0.5):
       validate_interp(args, device, model, scale, save_image=True, verbose=True)

@torch.inference_mode()
def validate_interp(args, device, model, test_scale, save_image=False, verbose=False):
    print(f'**[scale] : {test_scale}')
    dataset = RealMultiscaleSRDataset(args.test_GT_path, [2,3,4], preload=False, test=True)
    loader = DataLoader(dataset, num_workers=1)
    if(save_image and not os.path.exists('output_images')): os.makedirs('output_images')

    save_dir = 'output_images/interflow/scale_{:.1f}'.format(test_scale)

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    if not os.path.exists('output_images/original'): 
        os.makedirs('output_images/original')
    model.eval()

    target_scale = test_scale

    for step, test_data in enumerate(loader):
        gt = test_data['GT'].to(device)
        low_res4 = test_data['LR4'].to(device)
        low_res3 = test_data['LR3'].to(device)
        low_res2 = test_data['LR2'].to(device)

        save_tensor_image(gt, os.path.join(save_dir, 'Canon_{:03d}_HR.png'.format(step+1)))

        output = model.interpolate_image(low_res2, 2, low_res4, 4, target_scale, cond_gt=gt)

        print('[*] image: {:03d},\t shape: {}'.format(step+1, gt.shape))
        print('low_res2: r {:.4f}\tg {:.4f} \tb {:.4f}'.format(low_res2[:, 0].mean(), low_res2[:, 1].mean(), low_res2[:, 2].mean()))
        print('low_res3: r {:.4f}\tg {:.4f} \tb {:.4f}'.format(low_res3[:, 0].mean(), low_res3[:, 1].mean(), low_res3[:, 2].mean()))
        print('low_res4: r {:.4f}\tg {:.4f} \tb {:.4f}'.format(low_res4[:, 0].mean(), low_res4[:, 1].mean(), low_res4[:, 2].mean()))
        print('output: r {:.4f}\tg {:.4f} \tb {:.4f}'.format(output[:, 0].mean(), output[:, 1].mean(), output[:, 2].mean()))
        print()

        save_tensor_image(output, os.path.join(save_dir, 'Canon_{:03d}_LR{}.png'.format(step+1, 3)))

        check_val = output.cpu().numpy().reshape(1,-1,3).mean(axis=1)
        assert (check_val > 0).all() and (check_val < 1).all()    

def save_tensor_image(tensor_img, img_name):
    output = tensor_img[0].cpu().numpy()
    output = np.clip(output, 0., 1.0)
    output = output.transpose(1,2,0)
    Image.fromarray((output*255).astype(np.uint8)).save(img_name)

