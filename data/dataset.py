import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import torchvision.transforms.functional as F
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_mode, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor()
        ])
        self.loader = loader
        self.mask_mode = mask_mode
        self.image_size = image_size
        self.mask_config = {}

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        mask_img = img * (1. - mask) + mask*torch.randn_like(img)

        ret['gt_image'] = img
        ret['cond_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, index=0):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox(**self.mask_config))
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h, w))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size, **self.mask_config)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size, **self.mask_config)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox(**self.mask_config))
            irregular_mask = brush_stroke_mask(self.image_size, **self.mask_config)
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return F.to_tensor(mask)