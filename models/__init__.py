import torch
from core.praser import init_obj

def create_model(**cfg_model):
    """ create_model """
    opt = cfg_model['opt']
    logger = cfg_model['logger']

    model_opt = opt['model']['which_model']
    model_opt['args'].update(cfg_model)
    model = init_obj(model_opt, logger, default_file_name='models.model', init_type='Model')

    return model

def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    net = init_obj(network_opt, logger, default_file_name='models.network', init_type='Network')

    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__, network_opt['args'].get('init_type', 'default')))
        net.init_weights()
    return net


def define_loss(logger, loss_opt):
    return init_obj(loss_opt, logger, default_file_name='models.loss', init_type='Loss')

def define_metric(logger, metric_opt):
    return init_obj(metric_opt, logger, default_file_name='models.metric', init_type='Metric')

def define_optimizer(networks, logger, optimizer_opt):
    return init_obj(optimizer_opt, logger, networks, given_module=torch.optim, init_type='Optimizer')

def define_scheduler(optimizers, logger, scheduler_opt):
    return init_obj(scheduler_opt, logger, optimizers, given_module=torch.optim.lr_scheduler, init_type='Scheduler')