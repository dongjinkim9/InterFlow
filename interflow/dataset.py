import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from pathlib import Path


# Use this for Canon data only
class RealMultiscaleSRDataset(Dataset):
    def __init__(self, GT_path, scale:list, preload=True, transform=None, dataset_size=-1, batch_size=1, test=False):
        self.GT_path = GT_path
        self.test = test
        self.base_paths = [os.path.join(self.GT_path, str(i)) for i in [2, 3, 4]]
        self.preload = preload
        self.transform = transform
        self.dataset_size = dataset_size
        self.img_index = 0
        #self.real_size = len(os.listdir(self.base_paths[0])) // 2
        self.real_sizes = [len(list(filter(lambda x: x[-6:]=='HR.png', os.listdir(self.base_paths[i])))) for i in range(3)]
        #print(self.real_sizes, flush=True)
        self.scale = scale
        self.index_count = 0

        if preload:
            self.imgs = dict()
            for base_path in self.base_paths:
                for img_path in Path(base_path).glob('*.png'):
                    img_path = str(img_path)
                    self.imgs[img_path] = np.array(Image.open(img_path).convert('RGB'))


    def __len__(self):
        if(self.dataset_size == -1):
            return self.real_sizes[0]
        else:
            return self.dataset_size
        

    def __getitem__(self, trash_index):
        self.index_count += 1

        if(self.test):
            img_index = self.index_count
        else:
            img_index = random.randint(1, self.real_sizes[0])

        GT_img_path = os.path.join(self.base_paths[0], 'Canon_{:03d}_HR.png'.format(img_index))
        LR2_img_path = os.path.join(self.base_paths[0], 'Canon_{:03d}_LR{}.png'.format(img_index, 2))
        LR3_img_path = os.path.join(self.base_paths[1], 'Canon_{:03d}_LR{}.png'.format(img_index, 3))
        LR4_img_path = os.path.join(self.base_paths[2], 'Canon_{:03d}_LR{}.png'.format(img_index, 4))
        
        if self.preload:
            GT, LR1, LR2, LR3, LR4 = self.imgs[GT_img_path], self.imgs[GT_img_path], self.imgs[LR2_img_path], self.imgs[LR3_img_path], self.imgs[LR4_img_path]
        else:
            GT = np.array(Image.open(GT_img_path).convert('RGB'))
            LR1 = np.array(Image.open(GT_img_path).convert('RGB'))
            LR2 = np.array(Image.open(LR2_img_path).convert('RGB'))
            LR3 = np.array(Image.open(LR3_img_path).convert('RGB'))
            LR4 = np.array(Image.open(LR4_img_path).convert('RGB'))

        if(self.transform is not None):
            for tr in self.transform:
                GT, LR1, LR2, LR3, LR4 = tr(GT, GT, LR2, LR3, LR4)
        else:
            GT = shave_on_four(GT)
            LR1 = shave_on_four(LR1)
            LR2 = shave_on_four(LR2)
            LR3 = shave_on_four(LR3)
            LR4 = shave_on_four(LR4)

        img_item = {}
        img_item['GT'] = GT.transpose(2, 0, 1).astype(np.float32) / 255.    # This is fed into the non-invertible G function
        img_item['LR1'] = GT.transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR2'] = LR2.transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR3'] = LR3.transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR4'] = LR4.transpose(2, 0, 1).astype(np.float32) / 255.

        random_scale = random.choice(self.scale)
        img_item['scale'] = random_scale
        img_item['LR'] = img_item['LR{}'.format(random_scale)]

        if self.test:
            img_item['img_idx'] = img_index
        return img_item
   

class crop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, *inputs):
        ih, iw = inputs[0].shape[:2]
        try:
            ix = random.randrange(0, iw - self.patch_size +1)
            iy = random.randrange(0, ih - self.patch_size +1)
        except(ValueError):
            print('>> patch size: {}'.format(self.patch_size))
            print('>> ih, iw: {}, {}'.format(ih, iw))
            exit()

        output_list = [] 
        for inp in inputs:
            output_list.append(inp[iy : iy + self.patch_size, ix : ix + self.patch_size])
        
        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]


class augmentation(object):
    def __call__(self, *inputs):

        hor_flip = random.randrange(0,2)
        ver_flip = random.randrange(0,2)
        rot = random.randrange(0,2)

        output_list = []
        for inp in inputs:
            if hor_flip:
                tmp_inp = np.fliplr(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if ver_flip:
                tmp_inp = np.flipud(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if rot:
                inp = inp.transpose(1, 0, 2)
            output_list.append(inp)

        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]


def shave_on_four(img):
    shave = 8

    h, w, _ = img.shape
    if(h % shave != 0):
        img = img[:-(h%shave), :, :]
    if(w % shave != 0):
        img = img[:, :-(w%shave), :]
    return img


def limit_size(img, size_limit):
    h, w, _ = img.shape
    if(h > size_limit):
        img = img[:size_limit, :, :]
    if(w > size_limit):
        img = img[:, :size_limit, :]
    return img
