# Configure trainer and tester
# author: ynie
# date: Feb, 2020
from .training import Trainer
from .testing import Tester
from .dataloader import igibson_dataloader, rlsd_dataloader

def get_trainer(cfg, net, optimizer, device=None):
    return Trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)

def get_tester(cfg, net, device=None):
    return Tester(cfg=cfg, net=net, device=device)

def get_dataloader(config, mode):
    if config['data']['name'] == 'igibson':
        return igibson_dataloader(config=config, mode=mode)
    elif config['data']['name'] == 'rlsd':
        return rlsd_dataloader(config=config, mode=mode)
    elif config['data']['name'] == 's3d':
        return igibson_dataloader(config=config, mode=mode)
    else:
        raise NotImplementedError()