import os
from lpips import LPIPS, im2tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataset import RealSRDataset, RealSRpreloadDataset, crop, augmentation
from PIL import Image
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
import json
from json import JSONEncoder
from collections import defaultdict
from skimage.metrics import structural_similarity 
from model.registry import MODEL_REGISTRY
import model.models
import model.swinir
import wandb
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def train(args):
    # set logger
    wandb.init(project="InterFlow",
               config=args,
               name=args.parameter_name,
               notes=args.parameter_name,
               mode="disabled" if not args.wandb else None)
    artifact = wandb.Artifact(f'{args.parameter_name}_model', type='model',)
    
    writer = SummaryWriter('log/train_sr_networks')

    args.parameter_save_path = os.path.join(
        args.parameter_save_path, 
        f'iter{args.train_iteration}_{args.parameter_name}')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[*] Using GPU: {}'.format(args.gpu), flush=True)

    # Set dataset
    transform = [crop(args.patch_size), augmentation()]
    if args.preload:
        dataset = RealSRpreloadDataset(
            args.GT_paths,
            lazy_load=True, 
            transform=transform, 
            dataset_size=args.train_iteration*args.batch_size)
    else:
        dataset = RealSRDataset(
            args.GT_paths,
            lazy_load=True, 
            transform=transform, 
            dataset_size=args.train_iteration*args.batch_size)
    loader = DataLoader(
        dataset, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True)
    val_results = defaultdict(int)
    
    # get SR model
    model = MODEL_REGISTRY.get(args.model)()

    model = model.to(device)
    if(not args.parameter_restore_path is None):
        model.load_state_dict(torch.load(args.parameter_restore_path))
        print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))
    if(not args.parameter_save_path is None and not os.path.exists(args.parameter_save_path)):
        print(f'create folder: {args.parameter_save_path}')
        os.makedirs(args.parameter_save_path)
        
    model.train()

    # Set optimizer
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # set lr scheduler
    learning_rate_scheduler = MultiStepLR(
        optimizer, 
        milestones=[int(args.train_iteration * (1 - 1/(2**i))) for i in range(1,4)], 
        gamma=0.5)

    # save config data
    config = {'optimizer' : optimizer.state_dict(),
              'scheduler' : learning_rate_scheduler.state_dict(),
              'args' : args,}
    with open(os.path.join(args.parameter_save_path, "config.json"), "w") as json_file:
        json.dump(config, json_file, cls=MyEncoder)

    with tqdm(enumerate(loader)) as pbar:
        for step, train_data in pbar:
            gt = train_data['GT'].to(device)
            low_res = train_data['LR'].to(device)

            output = model(low_res)
            loss = l1_loss(gt, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()

            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            if (step+1) % args.parameter_save_iter == 0 and args.parameter_save_path is not None:
                save_path = os.path.join(args.parameter_save_path, f'iter{step+1}_{args.parameter_name}')
                torch.save(model.state_dict(), save_path)
            if (step+1) % args.validation_step == 0:
                val_results = validation(args, device, model, train_iter=step)
                wandb.log({'validation/canon_psnr': val_results['canon_psnr'],
                        'validation/canon_lpips': val_results['canon_lpips'],
                        'validation/canon_ssim': val_results['canon_ssim'],
                        'validation/nikon_psnr': val_results['nikon_psnr'],
                        'validation/nikon_lpips': val_results['nikon_lpips'],
                        'validation/nikon_ssim': val_results['nikon_ssim'],
                        'validation/total_psnr': val_results['total_psnr'],
                        'validation/total_lpips': val_results['total_lpips'],
                        'validation/total_ssim': val_results['total_ssim'],
                        'iter': step,})

                writer.add_scalar('validation/canon_psnr', val_results['canon_psnr'], step)
                writer.add_scalar('validation/canon_lpips', val_results['canon_lpips'], step)
                writer.add_scalar('validation/canon_ssim', val_results['canon_ssim'], step)
                writer.add_scalar('validation/nikon_psnr', val_results['nikon_psnr'], step)
                writer.add_scalar('validation/nikon_lpips', val_results['nikon_lpips'], step)
                writer.add_scalar('validation/nikon_ssim', val_results['nikon_ssim'], step)
                writer.add_scalar('validation/total_psnr', val_results['total_psnr'], step)
                writer.add_scalar('validation/total_lpips', val_results['total_lpips'], step)
                writer.add_scalar('validation/total_ssim', val_results['total_ssim'], step)
            
            if (step+1) % 500 == 0: 
                wandb.log({'train/loss': loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'iter': step,})
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step)
    
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

@torch.inference_mode()
def validation(args, device, model, train_iter=None):
    model.eval()

    # create Dataset and DataLoader
    canon_dataset = RealSRDataset(args.test_canon_GT_path, lazy_load=True, test=True)
    nikon_dataset = RealSRDataset(args.test_nikon_GT_path, lazy_load=True, test=True)
    canon_loader = DataLoader(canon_dataset, num_workers=1, pin_memory=True)
    nikon_loader = DataLoader(nikon_dataset, num_workers=1, pin_memory=True)

    loss_fn_vgg = LPIPS(net='vgg').to(device)

    # create folder for saving output images
    new_dir = os.path.join(
        'output_images', 
        f'iter{args.train_iteration}_{args.parameter_name}')
    if(not os.path.exists(new_dir)): 
        os.makedirs(new_dir, exist_ok=True)
    if train_iter is not None:
        new_dir = os.path.join(new_dir, str(train_iter+1))
        if(not os.path.exists(new_dir)): 
            os.makedirs(new_dir, exist_ok=True)

    shave = 4

    results = dict()
    camera_type = {'canon': canon_loader, 'nikon': nikon_loader}

    for ct, loader in camera_type.items():
        psnr_list, lpips_list, ssim_list = [], [], []
        for step, test_data in tqdm(enumerate(loader)): 
            gt = test_data['GT'].to(device)
            low_res = test_data['LR'].to(device)

            if (res_w := gt.shape[-1] % 4) != 0:
                gt = gt[:,:,:,:-res_w]
                low_res = low_res[:,:,:,:-res_w]
            if (res_h := gt.shape[-2] % 4) != 0:
                gt = gt[:,:,:-res_h,:]
                low_res = low_res[:,:,:-res_h,:]

            if args.model == "SwinIR":
                window_size = 128
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = low_res.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                low_res = torch.cat([low_res, torch.flip(low_res, [2])], 2)[:, :, :h_old + h_pad, :]
                low_res = torch.cat([low_res, torch.flip(low_res, [3])], 3)[:, :, :, :w_old + w_pad]
                output = model(low_res)
                output = output[..., :h_old, :w_old]
            else:
                output = model(low_res)
            
            output = output[0].cpu().numpy()
            output = np.clip(output, 0., 1.0)
            output = output.transpose(1,2,0)
            gt = gt[0].cpu().numpy()
            gt = gt.transpose(1,2,0)

            if args.test or (train_iter+1) % args.train_iteration == 0:
                Image.fromarray((output*255).astype(np.uint8)).save(
                    os.path.join(new_dir, '{}_{:03d}_out.png'.format(ct, step + 1)))

            y_output = rgb2ycbcr(output)[shave:-shave, shave:-shave, :1]
            y_gt = rgb2ycbcr(gt)[shave:-shave, shave:-shave, :1]
            
            psnr = peak_signal_noise_ratio(y_output, y_gt, data_range=255)
            psnr_list.append(psnr)
            
            ssim = structural_similarity(
                np.squeeze(y_output, axis=2),np.squeeze(y_gt, axis=2), data_range=255)
            ssim_list.append(ssim)

            lpips_gt = im2tensor(gt*255).to(device)
            lpips_pred = im2tensor(output*255).to(device)
            lpips = loss_fn_vgg(lpips_gt, lpips_pred)
            lpips_list.append(lpips.cpu().reshape((-1,)).item())

        results[f'{ct}_psnr'] = np.mean(psnr_list)
        results[f'{ct}_lpips'] = np.mean(lpips_list)
        results[f'{ct}_ssim'] = np.mean(ssim_list)

    results['total_psnr'] = round((results['canon_psnr'] + results['nikon_psnr'])/2, 2) 
    results['total_lpips'] = round((results['canon_lpips'] + results['nikon_lpips'])/2, 4)
    results['total_ssim'] = round((results['canon_ssim'] + results['nikon_ssim'])/2, 4)

    model.train()
    return results

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set network
    model = MODEL_REGISTRY.get(args.model)()

    model.to(device)
    assert not args.parameter_restore_path is None, 'Need to set restore parameter path'
    model.load_state_dict(torch.load(args.parameter_restore_path))
    print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))

    results = validation(args, device, model)
    print('\n[*] Results:\n {}'.format(results))