import argparse
import os
import torch.multiprocessing as mp

import palette.core.praser as Praser

from cleanfid import fid
from palette.core.base_dataset import BaseDataset
from palette.models.metric import inception_score

from palette import main_worker

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='phase', help='sub-command help')

parser_train = subparsers.add_parser('train')
parser_train.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
parser_train.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
parser_train.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser_train.add_argument('-d', '--debug', action='store_true')
parser_train.add_argument('-P', '--port', default='21012', type=str)

parser_test = subparsers.add_parser('test')
parser_test.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
parser_test.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
parser_test.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser_test.add_argument('-d', '--debug', action='store_true')
parser_test.add_argument('-P', '--port', default='21012', type=str)

parser_eval = subparsers.add_parser('eval')
parser_eval.add_argument('-s', '--src', type=str, help='Ground truth images directory')
parser_eval.add_argument('-d', '--dst', type=str, help='Generate images directory')

''' parser configs '''
args = parser.parse_args()

if args.phase in ('train', 'test'):
    opt = Praser.parse(args)
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)
        
elif args.phase == 'eval':
    fid_score = fid.compute_fid(args.src, args.dst)
    is_mean, is_std = inception_score(BaseDataset(args.dst), cuda=True, batch_size=8, resize=True, splits=10)
    
    print('FID: {}'.format(fid_score))
    print('IS:{} {}'.format(is_mean, is_std))

else:
    parser.print_help()