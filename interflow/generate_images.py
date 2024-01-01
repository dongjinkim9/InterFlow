import os
import torch
from torch.utils.data import DataLoader
from dataset import RealMultiscaleSRDataset
from models.flow_model import InterpFlow
from PIL import Image
import numpy as np
import argparse
from tqdm.auto import tqdm
from os import path as osp
from queue import Queue
import threading

def save_tensor_image(tensor_img:torch.Tensor, img_name:list):
    """
    tensor_img: (N,H,W,C)
    """
    output = tensor_img.cpu().numpy()
    output = np.clip(output, 0., 1.0)
    output = output.transpose(0,2,3,1)
    for img,img_path in zip(output,img_name):
        Image.fromarray((img*255).astype(np.uint8)).save(img_path)

class ImageSaver(threading.Thread):
    def __init__(self, shared_queue):
        threading.Thread.__init__(self)
        self.queue = shared_queue
    
    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            # do_work(item)
            img_path, img = item
            self.save_img(img_path, img)
            self.queue.task_done()
    
    def put_data(self,img_path,img):
        self.queue.put((img_path,img))

    def save_img(self, img_path, img):
        Image.fromarray(img).save(img_path)

@torch.inference_mode()
def generate_images(args, model, linear_weights):
    threads = []
    shared_queue = Queue()
    num_worker_threads = args.num_worker_threads

    for i in range(num_worker_threads):
        t = ImageSaver(shared_queue=shared_queue)
        t.start()
        threads.append(t)

    assert len(args.target_scales) == 2
    target_scales = sorted(args.target_scales)
    print(f'{target_scales=}')

    for idx, interp_coef in tqdm(enumerate(linear_weights)):
        dataset = RealMultiscaleSRDataset(args.dataset_dir, [4], preload=args.preload, test=True)
        loader = DataLoader(dataset, num_workers=1, batch_size=args.batch_size)
        
        target_name = scales[idx]
        print(f'{target_name=} {interp_coef=}')

        if(not osp.exists(args.save_dir)): os.makedirs(args.save_dir)
        save_dir = osp.join(args.save_dir, f'{folder_name}/scale_{target_name:.1f}')
        if(not osp.exists(save_dir)): os.makedirs(save_dir)

        
        for i, test_data in tqdm(enumerate(loader)):
            gt = test_data['GT'].to(device)
            low_res4 = test_data['LR4'].to(device)
            low_res3 = test_data['LR3'].to(device)
            low_res2 = test_data['LR2'].to(device)

            img_candidates = [gt,low_res2,low_res3,low_res4,]
            output = model.weighted_interpolate_image(
                img_candidates[target_scales[0]-1], img_candidates[target_scales[1]-1], 
                interp_coef, add_noise=False, cond_gt=gt)

            check_val = output.cpu().numpy().reshape(1,-1,3).mean(axis=1)

            if (check_val < 0).all() and (check_val > 1).all():
                tqdm.set_description('courrupted image generated! skipped.')
                continue

            output = output.cpu().numpy()
            output = (np.clip(output, 0., 1.0)*255).astype(np.uint8)
            output = output.transpose(0,2,3,1)
            gt = gt.cpu().numpy()
            gt = (np.clip(gt, 0., 1.0)*255).astype(np.uint8)
            gt = gt.transpose(0,2,3,1)

            label = test_data['img_idx']
            for lb,o,g in zip(label,output,gt):
                o_path = os.path.join(save_dir, f'Canon_{lb.item():03d}_LR{3}.png')
                g_path = os.path.join(save_dir, f'Canon_{lb.item():03d}_HR.png')
                shared_queue.put((o_path, o))
                shared_queue.put((g_path, g))

    # block until all tasks are done
    shared_queue.join()

    # stop workers
    for i in range(num_worker_threads):
        shared_queue.put(None)
    for t in threads:
        t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', type=str)
    parser.add_argument('--parameter', type=str)
    parser.add_argument('--dataset-dir', type=str, default='datasets/RealSR_v2_ordered/Nikon/Train_divided')
    parser.add_argument('--save-dir', type=str, default='output_images')
    parser.add_argument("--model-scales", type=int, nargs="+")
    parser.add_argument("--target-scales", type=int, nargs="+")
    parser.add_argument("--divide-mode", type=int, default=0)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-worker-threads', type=int, default=4)
    parser.add_argument('--preload', action='store_true', default=False)

    args = parser.parse_args()

    parameter_restore_path = args.parameter
    folder_name = args.foldername
    gpu = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InterpFlow(scales=args.model_scales,)
    model = model.to(device)
    model.load_state_dict(torch.load(parameter_restore_path))
    model.eval()

    ## for x2~x4
    if args.divide_mode == 0:
        scales = [2.1 + i*0.1 for i in range(19)]
        linear_weights = np.linspace(0.0, 1.0, 21)[1:-1]
    ## for x2~x3, and x3~x4
    elif args.divide_mode == 1:
        scales = [2.1 + i*0.1 for i in range(9)]
        linear_weights = np.linspace(0.0, 1.0, 11)[1:-1]
    elif args.divide_mode == 2:
        scales = [3.1 + i*0.1 for i in range(9)]
        linear_weights = np.linspace(0.0, 1.0, 11)[1:-1]
    assert len(scales) == len(linear_weights)

    generate_images(args, model, linear_weights)