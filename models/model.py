import re
import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, optimizers, lr_schedulers, losses, sample_num, task, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        ''' ddp '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        
        self.schedulers = lr_schedulers
        self.optG = optimizers[0]

        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.load_everything()
        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], writer=self.writer, phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in losses], *[m.__name__ for m in self.metrics], writer=self.writer, phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in losses], *[m.__name__ for m in self.metrics], writer=self.writer, phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.path = data['path']
    
    def get_current_visuals(self):
        dict = {
            'gt_image': (self.gt_image.detach()[:2].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:2].float().cpu()+1)/2,
            'mask': self.mask.detach()[:2].float().cpu()
        }
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('In_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.output[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.output[idx-self.batch_size].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            print('\r'+str(self.batch_size)+' '+str(self.iter), end=' ')
            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter % self.ema_scheduler['ema_iter'] == 0 and self.iter > self.ema_scheduler['ema_start']:
                    self.logger.info('Update the EMA  model at the iter {:.0f}'.format(self.iter))
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    self.val_metrics.update(met.__name__, met(self.cond_image, self.output))
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        for phase_data in tqdm.tqdm(self.phase_loader):
            self.set_input(phase_data)
            if self.opt['distributed']:
                if self.task in ['inpainting','uncropping']:
                    self.output = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                        y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                else:
                    self.output = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
            else:
                if self.task in ['inpainting','uncropping']:
                    self.output = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                        y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                else:
                    self.output = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                    
            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='test')
            for met in self.metrics:
                self.test_metrics.update(met.__name__, met(self.cond_image, self.output))
            for key, value in self.get_current_visuals().items():
                self.writer.add_images(key, value)
            self.writer.save_images(self.save_current_results())

    def load_everything(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)
        self.resume_training([self.optG], self.schedulers) 

    def save_everything(self):
        """ load pretrained model and training state, optimizers and schedulers must be a list. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state([self.optG], self.schedulers)
