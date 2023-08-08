import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from glob import glob
from scipy.stats import norm
from pathlib import Path
from torchvision import transforms
import torch

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
            
            # gt = Image.open(GT_img_path).convert('RGB')
            # LR2 = np.array(gt.resize((gt.width // 2.1, gt.height // 2.1),3).resize((gt.width, gt.height),3))
            # LR3 = np.array(gt.resize((gt.width // 3, gt.height // 3),3).resize((gt.width, gt.height),3))
            # LR4 = np.array(gt.resize((gt.width // 3.75, gt.height // 3.75),3).resize((gt.width, gt.height),3))
            LR2 = np.array(Image.open(LR2_img_path).convert('RGB'))
            LR3 = np.array(Image.open(LR3_img_path).convert('RGB'))
            LR4 = np.array(Image.open(LR4_img_path).convert('RGB'))

        if(self.transform is not None):
            for tr in self.transform:
                GT, LR1, LR2, LR3, LR4 = tr(GT, GT, LR2, LR3, LR4)
        else:
            # Avoid CUDA out of memory..
            # GT = limit_size(GT, 500)
            # LR2 = limit_size(LR2, 500)
            # LR3 = limit_size(LR3, 500)
            # LR4 = limit_size(LR4, 500)
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

        # bicubic
        # GT_pil = Image.fromarray(GT)
        # img_item['LR5'] = np.array(GT_pil.resize((int(GT_pil.width // 2.1), int(GT_pil.height // 2.1)),3).resize((GT_pil.width, GT_pil.height),3)).transpose(2, 0, 1).astype(np.float32) / 255.
        # img_item['LR6'] = np.array(GT_pil.resize((int(GT_pil.width // 3.0), int(GT_pil.height // 3.0)),3).resize((GT_pil.width, GT_pil.height),3)).transpose(2, 0, 1).astype(np.float32) / 255.
        # img_item['LR7'] = np.array(GT_pil.resize((int(GT_pil.width // 3.75), int(GT_pil.height // 3.75)),3).resize((GT_pil.width, GT_pil.height),3)).transpose(2, 0, 1).astype(np.float32) / 255.

        #random_scale = random.randint(2, 4)
        random_scale = random.choice(self.scale)
        img_item['scale'] = random_scale
        img_item['LR'] = img_item['LR{}'.format(random_scale)]

        if self.test:
            img_item['img_idx'] = img_index
        return img_item

# Use this for Canon data only
class RealSRDataset(Dataset):
    """
    dataset path naming convention:
    {$base_path}/{$camera_type}/{'Train'|'Test'}/{$scale}
    $camera_type : 'Canon' or 'Nikon'  
    """
    def __init__(
        self, 
        GT_paths,
        sampling_method = 'uniform',
        sampling_weight = None,
        lazy_load=True, 
        transform=None, 
        dataset_size=-1, 
        test=False,
        normalized = True,
        ):
        # if scales input get float type, convert it to list
        if isinstance(GT_paths, str):
            self.GT_paths = []
            self.GT_paths.append(GT_paths)
        else:
            self.GT_paths = GT_paths

        self.test = test
        self.lazy_load = lazy_load
        self.transform = transform
        self.dataset_size = dataset_size
        self.img_index = 0
        self.real_sizes = [len(glob(os.path.join(GT_path, '*HR.png'))) for GT_path in self.GT_paths]
        self.index_count = 0
        self.sampling_weight = []
        self.normalized = normalized

        self.population=[_ for _ in range(len(self.GT_paths))]
        if sampling_method == "uniform":
            self.sampling_weight = [1 for _ in range(len(self.GT_paths))]
        elif sampling_method == "normal":
            mean, sigma = 0, 1
            num_paths = len(self.GT_paths)
            x_list = [round((-3*sigma) + 6*sigma*(1.0*n/(num_paths+1)),5) for n in range(1,num_paths+1)] 
            self.sampling_weight = [norm(mean, sigma).pdf(x) for x in x_list]
        elif sampling_method == "custom":
            assert sampling_weight != None and len(sampling_weight) == len(self.GT_paths)
            self.sampling_weight = sampling_weight
        else:
            raise NotImplementedError()
        assert all(self.real_sizes), 'incorrect path! cannot find any image in the folder'

    def __len__(self):
        if(self.dataset_size == -1):
            return self.real_sizes[0]
        else:
            return self.dataset_size
        

    def __getitem__(self, trash_index):
        self.index_count += 1
        # GT_paths_index = random.randint(0, len(self.GT_paths)-1)
        GT_paths_index = random.choices(
            population=self.population,
            weights=self.sampling_weight) 
        assert len(GT_paths_index) == 1
        base_path = self.GT_paths[GT_paths_index[0]]

        if(self.test):
            img_index = self.index_count
        else:
            img_index = random.randint(1, self.real_sizes[GT_paths_index[0]])

        GT_img_path = os.path.join(base_path, 'Canon_{:03d}_HR.png'.format(img_index))
        lr_cadidate = glob(os.path.join(base_path, 'Canon_{:03d}_*.png'.format(img_index)))
        lr_candidate_filtered = list(filter(lambda x: not 'HR' in x, lr_cadidate))
        assert len(lr_candidate_filtered) == 1, 'No LR candidate in your dataset'
        LR_img_path = lr_candidate_filtered[0]
        
        GT = np.array(Image.open(GT_img_path).convert('RGB'))
        LR = np.array(Image.open(LR_img_path).convert('RGB'))

        if(self.transform is not None):
            for tr in self.transform:
                GT, LR = tr(GT, LR)

        img_item = {}
        img_item['GT'] = GT.transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = LR.transpose(2, 0, 1).astype(np.float32)
        if self.normalized:
            img_item['GT'] = img_item['GT'] / 255.
            img_item['LR'] = img_item['LR'] / 255.
        
        return img_item
        # return GT_img_path, LR_img_path

class RealSRpreloadDataset(Dataset):
    """
    dataset path naming convention:
    {$base_path}/{$camera_type}/{'Train'|'Test'}/{$scale}
    $camera_type : 'Canon' or 'Nikon'  
    """
    def __init__(
        self, 
        GT_paths,
        sampling_method = 'uniform',
        sampling_weight = None,
        lazy_load=True, 
        transform=None, 
        dataset_size=-1, 
        test=False,
        normalized = True,
        ):
        # if scales input get float type, convert it to list
        if isinstance(GT_paths, str):
            self.GT_paths = []
            self.GT_paths.append(GT_paths)
        else:
            self.GT_paths = GT_paths

        self.test = test
        self.lazy_load = lazy_load
        self.transform = transform
        self.dataset_size = dataset_size
        self.img_index = 0
        self.real_sizes = [len(glob(os.path.join(GT_path, '*HR.png'))) for GT_path in self.GT_paths]
        self.index_count = 0
        self.sampling_weight = []
        self.normalized = normalized

        self.population=[_ for _ in range(len(self.GT_paths))]
        if sampling_method == "uniform":
            self.sampling_weight = [1 for _ in range(len(self.GT_paths))]
        elif sampling_method == "normal":
            mean, sigma = 0, 1
            num_paths = len(self.GT_paths)
            x_list = [round((-3*sigma) + 6*sigma*(1.0*n/(num_paths+1)),5) for n in range(1,num_paths+1)] 
            self.sampling_weight = [norm(mean, sigma).pdf(x) for x in x_list]
        elif sampling_method == "custom":
            assert sampling_weight != None and len(sampling_weight) == len(self.GT_paths)
            self.sampling_weight = sampling_weight
        else:
            raise NotImplementedError()
        assert all(self.real_sizes), f'{self.real_sizes=} \n incorrect path! cannot find any image in the folder'

        self.imgs = dict()
        for base_path in self.GT_paths:
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
        # GT_paths_index = random.randint(0, len(self.GT_paths)-1)
        GT_paths_index = random.choices(
            population=self.population,
            weights=self.sampling_weight) 
        assert len(GT_paths_index) == 1
        base_path = self.GT_paths[GT_paths_index[0]]

        if(self.test):
            img_index = self.index_count
        else:
            img_index = random.randint(1, self.real_sizes[GT_paths_index[0]])

        # GT_img_path = Path(base_path) / f'Canon_{img_index:03d}_HR.png'
        # LR_img_path = Path(base_path) / f'Canon_{img_index:03d}_LR.png'
        GT_img_path = os.path.join(base_path, 'Canon_{:03d}_HR.png'.format(img_index))
        lr_cadidate = glob(os.path.join(base_path, 'Canon_{:03d}_*.png'.format(img_index)))
        lr_candidate_filtered = list(filter(lambda x: not 'HR' in x, lr_cadidate))
        assert len(lr_candidate_filtered) == 1, 'No LR candidate in your dataset'
        LR_img_path = lr_candidate_filtered[0]
        
        GT = self.imgs[str(GT_img_path)]
        LR = self.imgs[str(LR_img_path)]

        if(self.transform is not None):
            for tr in self.transform:
                GT, LR = tr(GT, LR)

        img_item = {}

        img_item['GT'] = GT.transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = LR.transpose(2, 0, 1).astype(np.float32)
        if self.normalized:
            img_item['GT'] = img_item['GT'] / 255.
            img_item['LR'] = img_item['LR'] / 255.

        return img_item
        # return GT_img_path, LR_img_path

class DRealSRTestDataset(Dataset):
    def __init__(self, dataset_path, transform=None, preload=False):
        base_pth = Path(dataset_path)
        self.GT_paths = sorted([str(pth) for pth in (base_pth / 'test_HR').glob('*.png')])
        self.LQ_paths = sorted([str(pth) for pth in (base_pth / 'test_LR').glob('*.png')])
        self.transform = transform
        self.preload = preload

        assert len(self.GT_paths) != 0
        for pths in zip(self.GT_paths, self.LQ_paths):
            assert pths[0].split('_')[1] == pths[1].split('_')[1]

        if self.preload:
            self.GTs = [np.array(Image.open(img_path).convert('RGB')) for img_path in self.GT_paths]
            self.LQs = [np.array(Image.open(img_path).convert('RGB')) for img_path in self.LQ_paths]


    def __len__(self):
        return len(self.GT_paths)
        

    def __getitem__(self, idx):
        if self.preload:
            GT = self.GTs[idx]
            LQ = self.LQs[idx]
        else:
            GT = np.array(Image.open(self.GT_paths[idx]).convert('RGB'))
            LQ = np.array(Image.open(self.LQ_paths[idx]).convert('RGB'))

        if(self.transform is not None):
            for tr in self.transform:
                GT, LQ = tr(GT, LQ)

        img_item = {}
        img_item['GT'] = GT.transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR'] = LQ.transpose(2, 0, 1).astype(np.float32) / 255.

        img_item['filename'] = Path(self.GT_paths[idx]).name

        return img_item

def shave_on_four(img):
    h, w, _ = img.shape
    if(h % 4 != 0):
        img = img[:-(h%4), :, :]
    if(w % 4 != 0):
        img = img[:, :-(w%4), :]
    return img


def limit_size(img, size_limit):
    h, w, _ = img.shape
    if(h > size_limit):
        img = img[:size_limit, :, :]
    if(w > size_limit):
        img = img[:, :size_limit, :]
    return img

####

# https://github.com/yinboc/liif/blob/main/datasets/wrappers.py

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC, antialias=True)(
            transforms.ToPILImage()(img)))

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

class LiifRealSRDataset(Dataset):
    """
    dataset path naming convention:
    {$base_path}/{$camera_type}/{'Train'|'Test'}/{$scale}
    $camera_type : 'Canon' or 'Nikon'  
    """
    def __init__(
        self, 
        GT_paths,
        sampling_method = 'uniform',
        sampling_weight = None,
        preload=False, 
        transform=None, 
        dataset_size=-1, 
        test=False,
        normalized = True,
        ):
        # if scales input get float type, convert it to list
        if isinstance(GT_paths, str):
            self.GT_paths = []
            self.GT_paths.append(GT_paths)
        else:
            self.GT_paths = GT_paths

        self.test = test
        self.transform = transform
        self.dataset_size = dataset_size
        self.img_index = 0
        self.real_sizes = [len(glob(os.path.join(GT_path, '*HR.png'))) for GT_path in self.GT_paths]
        self.index_count = 0
        self.sampling_weight = []
        self.normalized = normalized
        self.preload = preload

        self.population=[_ for _ in range(len(self.GT_paths))]
        if sampling_method == "uniform":
            self.sampling_weight = [1 for _ in range(len(self.GT_paths))]
        elif sampling_method == "normal":
            mean, sigma = 0, 1
            num_paths = len(self.GT_paths)
            x_list = [round((-3*sigma) + 6*sigma*(1.0*n/(num_paths+1)),5) for n in range(1,num_paths+1)] 
            self.sampling_weight = [norm(mean, sigma).pdf(x) for x in x_list]
        elif sampling_method == "custom":
            assert sampling_weight != None and len(sampling_weight) == len(self.GT_paths)
            self.sampling_weight = sampling_weight
        else:
            raise NotImplementedError()
        assert all(self.real_sizes), 'incorrect path! cannot find any image in the folder'
        
        if preload:
            self.imgs = dict()
            for base_path in self.GT_paths:
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
        # GT_paths_index = random.randint(0, len(self.GT_paths)-1)
        GT_paths_index = random.choices(
            population=self.population,
            weights=self.sampling_weight) 
        assert len(GT_paths_index) == 1
        base_path = self.GT_paths[GT_paths_index[0]]

        if(self.test):
            img_index = self.index_count
        else:
            img_index = random.randint(1, self.real_sizes[GT_paths_index[0]])

        # GT_img_path = Path(base_path) / f'Canon_{img_index:03d}_HR.png'
        # LR_img_path = Path(base_path) / f'Canon_{img_index:03d}_LR.png'
        GT_img_path = os.path.join(base_path, 'Canon_{:03d}_HR.png'.format(img_index))
        lr_cadidate = glob(os.path.join(base_path, 'Canon_{:03d}_*.png'.format(img_index)))
        lr_candidate_filtered = list(filter(lambda x: not 'HR' in x, lr_cadidate))
        assert len(lr_candidate_filtered) == 1, 'No LR candidate in your dataset'
        LR_img_path = lr_candidate_filtered[0]
        
        if self.preload:
            GT = self.imgs[str(GT_img_path)]
            LR = self.imgs[str(LR_img_path)]
        else:
            GT = np.array(Image.open(GT_img_path).convert('RGB'))
            LR = np.array(Image.open(LR_img_path).convert('RGB'))


        if(self.transform is not None):
            for tr in self.transform:
                GT, LR = tr(GT, LR)

        img_item = {}

        img_item['GT'] = torch.from_numpy(GT.transpose(2, 0, 1).astype(np.float32))
        img_item['LR'] = torch.from_numpy(LR.transpose(2, 0, 1).astype(np.float32))
        if self.normalized:
            img_item['GT'] = img_item['GT'] / 255.
            img_item['LR'] = img_item['LR'] / 255.

        return img_item

class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, only_gt=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.only_gt = only_gt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # img = self.dataset[idx]
        batch = self.dataset[idx]
        img_lr, img_hr = batch['LR'], batch['GT']
        
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            raise NotImplementedError()
            # h_lr = math.floor(img.shape[-2] / s + 1e-9)
            # w_lr = math.floor(img.shape[-1] / s + 1e-9)
            # img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            # img_down = resize_fn(img, (h_lr, w_lr))
            # crop_lr, crop_hr = img_down, img
        else:
            # w_lr = self.inp_size
            # w_hr = round(w_lr * s)
            # x0 = random.randint(0, img.shape[-2] - w_hr)
            # y0 = random.randint(0, img.shape[-1] - w_hr)
            # crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            # crop_lr = resize_fn(crop_hr, w_lr)

            w_lr = self.inp_size
            # hr and lr have same res
            w_hr = round(w_lr * s)
            # w_hr = self.inp_size
            
            x0 = random.randint(0, img_hr.shape[-2] - w_hr)
            y0 = random.randint(0, img_hr.shape[-1] - w_hr)
            crop_hr = img_hr[:, x0: x0 + w_hr, y0: y0 + w_hr]
            if self.only_gt:
                crop_lr = resize_fn(crop_hr, w_lr)
            else:
                crop_lr_temp = img_lr[:, x0: x0 + w_hr, y0: y0 + w_hr]
                crop_lr = resize_fn(crop_lr_temp, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            # 'full_gt': crop_hr,
        }


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

class crop_ratio(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def set_ratio(self, ratio):
        self.img_ratio = ratio

    def __call__(self, GT, LQ):
        assert (np.array(GT.shape[:2]) == np.floor(np.array(LQ.shape[:2]) * self.img_ratio)).all(), f'{GT.shape[:2]=} != {LQ.shape[:2]=}'
        assert (np.array(LQ.shape[:2]) > 48).all()
        LQ_ih, LQ_iw = LQ.shape[:2]
        try:
            LQ_ix = random.randrange(0, LQ_iw - self.patch_size +1)
            LQ_iy = random.randrange(0, LQ_ih - self.patch_size +1)
        except(ValueError):
            print('>> patch size: {}'.format(self.patch_size))
            print('>> ih, iw: {}, {}'.format(LQ_ih, LQ_iw))
            exit()

        GT_ix, GT_iy = int(LQ_ix * self.img_ratio), int(LQ_iy * self.img_ratio)
        LQ_ix, LQ_iy = int(LQ_ix), int(LQ_iy)
        # print(GT_ix, self.patch_size)
        output_list = [] 
        output_list.append(GT[GT_iy : GT_iy + int(self.patch_size*self.img_ratio), GT_ix : GT_ix + int(self.patch_size*self.img_ratio)])
        output_list.append(LQ[LQ_iy : LQ_iy + self.patch_size, LQ_ix : LQ_ix + self.patch_size])
        
        return output_list

        # if(len(output_list) > 1):
        #     return output_list
        # else:
        #     return output_list[0]

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

class LiifRealSRMixupDataset(Dataset):
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

        self.inp_size = 48
        self.sample_q = 2304

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
        # LR3_img_path = os.path.join(self.base_paths[1], 'Canon_{:03d}_LR{}.png'.format(img_index, 3))
        LR4_img_path = os.path.join(self.base_paths[2], 'Canon_{:03d}_LR{}.png'.format(img_index, 4))
        
        if self.preload:
            # GT, LR1, LR2, LR3, LR4 = self.imgs[GT_img_path], self.imgs[GT_img_path], self.imgs[LR2_img_path], self.imgs[LR3_img_path], self.imgs[LR4_img_path]
            GT, LR2, LR4 = self.imgs[GT_img_path], self.imgs[LR2_img_path], self.imgs[LR4_img_path]
        else:
            GT = np.array(Image.open(GT_img_path).convert('RGB'))
            # LR1 = np.array(Image.open(GT_img_path).convert('RGB'))
            LR2 = np.array(Image.open(LR2_img_path).convert('RGB'))
            # LR3 = np.array(Image.open(LR3_img_path).convert('RGB'))
            LR4 = np.array(Image.open(LR4_img_path).convert('RGB'))

        if(self.transform is not None):
            for tr in self.transform:
                # GT, LR1, LR2, LR3, LR4 = tr(GT, GT, LR2, LR3, LR4)
                GT, LR2, LR4 = tr(GT, LR2, LR4)
        else:
            GT = shave_on_four(GT)
            # LR1 = shave_on_four(LR1)
            LR2 = shave_on_four(LR2)
            # LR3 = shave_on_four(LR3)
            LR4 = shave_on_four(LR4)

        img_item = {}
        img_item['GT'] = torch.from_numpy(GT.transpose(2, 0, 1).astype(np.float32) / 255.)
        # img_item['LR1'] = torch.from_numpy(GT.transpose(2, 0, 1).astype(np.float32) / 255.)
        img_item['LR2'] = torch.from_numpy(LR2.transpose(2, 0, 1).astype(np.float32) / 255.)
        # img_item['LR3'] = torch.from_numpy(LR3.transpose(2, 0, 1).astype(np.float32) / 255.)
        img_item['LR4'] = torch.from_numpy(LR4.transpose(2, 0, 1).astype(np.float32) / 255.)

        random_scale = random.choice(self.scale)
        alpha = random.random()
        s = (1-alpha)*2.1 + alpha*3.75
        img_lr = (1-alpha)*img_item['LR2'] + alpha*img_item['LR4']
        img_hr = img_item['GT']

        w_lr = self.inp_size
        # hr and lr have same res
        w_hr = round(w_lr * s)
        # w_hr = self.inp_size
        
        x0 = random.randint(0, img_hr.shape[-2] - w_hr)
        y0 = random.randint(0, img_hr.shape[-1] - w_hr)
        crop_hr = img_hr[:, x0: x0 + w_hr, y0: y0 + w_hr]

        crop_lr_temp = img_lr[:, x0: x0 + w_hr, y0: y0 + w_hr]
        crop_lr = resize_fn(crop_lr_temp, w_lr)
        
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            # 'full_gt': crop_hr,
        }
