import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = list(np.genfromtxt(dir, dtype=np.str, encoding='utf-8'))
    else:
        images = []
        assert os.path.isdir(dir), f'{dir} is not a valid directory'
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config=None, data_len=-1, image_size=None, loader=pil_loader):
        if mask_config is None:
            mask_config = {}
        if image_size is None:
            image_size = [256, 256]
        imgs = make_dataset(data_root)
        self.imgs = imgs[:int(data_len)] if data_len > 0 else imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask
        return {
            'gt_image': img,
            'cond_image': cond_image,
            'mask_image': mask_img,
            'mask': mask,
            'path': path.rsplit("/")[-1].rsplit("\\")[-1]
        }

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):  # sourcery skip: remove-pass-elif
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config=None, data_len=-1, image_size=None, loader=pil_loader):
        if mask_config is None:
            mask_config = {}
        if image_size is None:
            image_size = [256, 256]
        imgs = make_dataset(data_root)
        self.imgs = imgs[:int(data_len)] if data_len > 0 else imgs
        self.tfs = transforms.Compose([transforms.Resize((image_size[0], image_size[1])), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask
        return {
            'gt_image': img,
            'cond_image': cond_image,
            'mask_image': mask_img,
            'mask': mask,
            'path': path.rsplit("/")[-1].rsplit("\\")[-1]
        }

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):  # sourcery skip: remove-pass-elif
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode in ['fourdirection', 'onedirection']:
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0, 2) < 1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=None, loader=pil_loader):
        if image_size is None:
            image_size = [224, 224]
        self.data_root = data_root
        flist = make_dataset(data_flist)
        self.flist = flist[:int(data_len)] if data_len > 0 else flist
        self.tfs = transforms.Compose([transforms.Resize((image_size[0], image_size[1])), transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        file_name = f'{str(self.flist[index]).zfill(5)}.png'
        img = self.tfs(self.loader(f'{self.data_root}/color/{file_name}'))

        cond_image = self.tfs(self.loader(f'{self.data_root}/gray/{file_name}'))

        return {
            'gt_image': img,
            'cond_image': cond_image,
            'path': file_name
        }

    def __len__(self):
        return len(self.flist)
