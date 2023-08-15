import os
from lpips import LPIPS, im2tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataset import *
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
import json
from json import JSONEncoder
from collections import defaultdict
from skimage.metrics import structural_similarity 
from model.registry import MODEL_REGISTRY
import model.models
import wandb
from torch.utils.data import ConcatDataset
from dataset import resize_fn
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
    transform = [augmentation()]


    with open('./configs/arb_train_configs.json', "r") as confs:
        train_setting = json.load(confs)
    setting = train_setting[args.parameter_name]
    print(setting)
    dataset_list = [
        SRImplicitDownsampled(
            dataset=LiifRealSRDataset(GT_paths, preload=args.preload, transform=transform, 
                dataset_size=args.train_iteration*args.batch_size, normalized = True,),
            scale_min=scale[0],scale_max=scale[1],inp_size=48,augment=False,sample_q=2304,only_gt=only_gt,
        )
        for GT_paths, scale, only_gt in zip(setting['GT_paths'], setting['scale'], setting['only_gt'])
    ]

    dataset = ConcatDataset(dataset_list)

    loader = DataLoader(
        dataset, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True)
    val_results = defaultdict(int)
    
    # get SR model
    model = MODEL_REGISTRY.get(args.model)(encoder_type = args.encoder)

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
        for step, batch in pbar:
            if step >= args.train_iteration: break
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp, gt = batch['inp'], batch['gt']
            pred = model(inp, batch['coord'], batch['cell'])

            loss = l1_loss(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate_scheduler.step()

            if (step+1) % args.validation_step == 0:
                val_results = validate(args, device, model, train_iter=step)
            
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
    
            if (step+1) % args.checkpoint_step == 0 and args.parameter_save_path is not None:
                save_path = os.path.join(args.parameter_save_path, f'iter{step+1}_{args.parameter_name}')
                torch.save(model.state_dict(), save_path)
        
    if save_path:
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

def test(args):
    from pprint import pprint
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[*] Using GPU: {}'.format(args.gpu), flush=True)

    model = MODEL_REGISTRY.get(args.model)()

    model = model.to(device)
    if(not args.parameter_restore_path is None):
        model.load_state_dict(torch.load(args.parameter_restore_path))
        print('[*] pre-trained model is loaded from {}'.format(args.parameter_restore_path))

    if args.test_data == "RealSR":
        results = validate(args, device, model, train_iter=None)
    elif args.test_data == "DRealSR":
        results = validate_drealsr(args, device, model, train_iter=None)
    else:
        raise NotImplementedError()
    pprint(results)

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

@torch.inference_mode()
def validate(args, device, model, train_iter=None):
    model.eval()

    # create Dataset and DataLoader
    canon_dataset = LiifRealSRDataset(args.test_canon_GT_path, preload=False, test=True)
    nikon_dataset = LiifRealSRDataset(args.test_nikon_GT_path, preload=False, test=True)
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
            if (res_w := test_data['GT'].shape[-1] % 3) != 0:
                test_data['GT'] = test_data['GT'][:,:,:,:-res_w]
                test_data['LR'] = test_data['LR'][:,:,:,:-res_w]
            if (res_h := test_data['GT'].shape[-2] % 3) != 0:
                test_data['GT'] = test_data['GT'][:,:,:-res_h,:]
                test_data['LR'] = test_data['LR'][:,:,:-res_h,:]
                
            gt = test_data['GT'].to(device)
            low_res = test_data['LR'].to(device)

            h, w = gt.shape[-2], gt.shape[-1]

            low_res = resize_fn(low_res.squeeze(0), (int(h//args.test_scale), int(w//args.test_scale)))
            low_res = low_res.to(device)
            coord = make_coord((h, w)).to(device)
            cell = torch.ones_like(coord).to(device)
            cell[:, 0] *= 2 / h
            cell[:, 1] *= 2 / w
            pred = batched_predict(
                model, 
                low_res.unsqueeze(0),
                coord.unsqueeze(0), cell.unsqueeze(0), 
                bsize=30000)[0]
            pred = pred.clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
            
            if args.test or (train_iter+1) % args.train_iteration == 0:
                transforms.ToPILImage()(pred).save(os.path.join(new_dir, '{}_{:03d}_out.png'.format(ct, step + 1)))
                transforms.ToPILImage()(low_res).save(os.path.join(new_dir, '{}_{:03d}_LR.png'.format(ct, step + 1)))
            output = pred.numpy().transpose(1,2,0)
            gt = gt[0].cpu().numpy().transpose(1,2,0)

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

# Drealsr x3
@torch.inference_mode()
def validate_drealsr(args, device, model, train_iter=None):
    model.eval()

    # create Dataset and DataLoader
    assert len(args.GT_paths) == 1
    dataset = DRealSRTestDataset(args.GT_paths[0], preload=False)
    dataloader = DataLoader(dataset, num_workers=1, pin_memory=True)

    # loss_fn_vgg = LPIPS(net='vgg').to(device)
    loss_fn_vgg = LPIPS(net='vgg')

    # create folder for saving output images
    new_dir = os.path.join('output_images', f'{args.parameter_name}')
    if not os.path.exists(new_dir): 
        os.makedirs(new_dir, exist_ok=True)

    shave = 4

    psnr_list, lpips_list, ssim_list = defaultdict(list), defaultdict(list), defaultdict(list)

    results = dict()
    for step, test_data in tqdm(enumerate(dataloader)):
        if (res_w := test_data['GT'].shape[-1] % 3) != 0:
            test_data['GT'] = test_data['GT'][:,:,:,:-res_w]
            test_data['LR'] = test_data['LR'][:,:,:,:-res_w]
        if (res_h := test_data['GT'].shape[-2] % 3) != 0:
            test_data['GT'] = test_data['GT'][:,:,:-res_h,:]
            test_data['LR'] = test_data['LR'][:,:,:-res_h,:]
            
        gt = test_data['GT'].to(device)
        low_res = test_data['LR'].to(device)

        ct = test_data['filename'][0].split('_')[0][0]
        if ct =='D':
            ct = 'Nikon'
        elif ct =='I':
            ct = 'Canon'
        elif ct =='P':
            ct = 'Olympus'
        elif ct =='s':
            ct = 'Sony'
        elif ct =='p':
            ct = 'Panasonic'

        h, w = gt.shape[-2], gt.shape[-1]

        low_res = resize_fn(low_res.squeeze(0), (h // 3, w //3))
        low_res = low_res.to(device)
        coord = make_coord((h, w)).to(device)
        cell = torch.ones_like(coord).to(device)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        pred = batched_predict(model, low_res.unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = pred.clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        
        transforms.ToPILImage()(pred).save(os.path.join(new_dir, '{}_{:03d}_out.png'.format(ct, step + 1)))
        transforms.ToPILImage()(low_res).save(os.path.join(new_dir, '{}_{:03d}_LR.png'.format(ct, step + 1)))

        if args.test or (train_iter+1) % args.train_iteration == 0:
            transforms.ToPILImage()(pred).save(os.path.join(new_dir, '{}_{:03d}_out.png'.format(ct, step + 1)))
            transforms.ToPILImage()(low_res).save(os.path.join(new_dir, '{}_{:03d}_LR.png'.format(ct, step + 1)))
        output = pred.numpy().transpose(1,2,0)
        gt = gt[0].cpu().numpy().transpose(1,2,0)

        y_output = rgb2ycbcr(output)[shave:-shave, shave:-shave, :1]
        y_gt = rgb2ycbcr(gt)[shave:-shave, shave:-shave, :1]
        
        psnr = peak_signal_noise_ratio(y_output, y_gt, data_range=255)
        psnr_list[ct].append(psnr)
        
        ssim = structural_similarity(np.squeeze(y_output, axis=2),np.squeeze(y_gt, axis=2), data_range=255)
        ssim_list[ct].append(ssim)

        lpips_gt = im2tensor(gt*255).to(device)
        lpips_pred = im2tensor(output*255).to(device)
        lpips = loss_fn_vgg(lpips_gt.cpu(), lpips_pred.cpu())
        lpips_list[ct].append(lpips.cpu().reshape((-1,)).item())

    for ct_name in psnr_list.keys():
        results[f'psnr_{ct_name}'] = np.mean(psnr_list[ct_name])
        results[f'ssim_{ct_name}'] = np.mean(ssim_list[ct_name])
        results[f'lpips_{ct_name}'] = np.mean(lpips_list[ct_name])
    results['total_psnr'] = round(np.stack([results[ct_name] for ct_name in results.keys() if 'psnr' in ct_name]).mean(), 2)
    results['total_ssim'] = round(np.stack([results[ct_name] for ct_name in results.keys() if 'ssim' in ct_name]).mean(), 4)
    results['total_lpips'] = round(np.stack([results[ct_name] for ct_name in results.keys() if 'lpips' in ct_name]).mean(), 4)

    model.train()
    return results