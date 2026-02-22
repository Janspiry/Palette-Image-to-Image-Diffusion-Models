import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    return Image.open(path).convert('RGB')

class VirtualStainingDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        
        # Définition des sous-dossiers en fonction du data_root (train, test ou valid)
        self.hes_dir = os.path.join(data_root, 'HES')
        self.cd30_dir = os.path.join(data_root, 'CD30')
        
        # On liste toutes les images basées sur le dossier HES
        if os.path.isdir(self.hes_dir):
            self.image_names = sorted([f for f in os.listdir(self.hes_dir) if is_image_file(f)])
        else:
            self.image_names = []
            
        if data_len > 0:
            self.image_names = self.image_names[:int(data_len)]
            
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = self.image_names[index]

        # 1. HES = Input (Conditionnement)
        hes_path = os.path.join(self.hes_dir, file_name)
        cond_image = self.tfs(self.loader(hes_path))

        # 2. CD30 = Ground Truth (Cible)
        cd30_path = os.path.join(self.cd30_dir, file_name)
        gt_image = self.tfs(self.loader(cd30_path))

        ret['gt_image'] = gt_image
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.image_names)