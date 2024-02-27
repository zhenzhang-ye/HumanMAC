from utils.evaluation import compute_stats
from utils.training import Trainer
from tensorboardX import SummaryWriter
import torch
from config import Config, update_config
import argparse
import sys

from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *
from omegaconf import DictConfig, OmegaConf
import hydra
from motpred_eval.motpred.utils.config import merge_cfg
from motpred_interface.motpred_prepare_humanmac import load_model_config_exp, prepare_model, set_updataset
from motpred_eval.motpred.eval_utils import prepare_eval_dataset
from utils.util import get_dct_matrix

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.append(os.getcwd())

OmegaConf.register_new_resolver("eval", eval)
@hydra.main(config_path="./motpred_interface/config", config_name="config")
def main_hydra(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # for backwards compatibility with the code
    for subconf in ['task', 'dataset']:
        if subconf in cfg:
            cfg = {**cfg, **cfg[subconf]}
            cfg.pop(subconf)
    train(cfg)  

def create_train_dataset(batch_size, data_loader_name, num_workers, shuffle=False, drop_last=False, **config):
    from motpred_eval.motpred.dataset import custom_collate_for_mmgt
    from torch.utils.data import DataLoader
    dataset = set_updataset(config, data_loader_name)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory= True,
                                num_workers=num_workers, drop_last=drop_last, collate_fn=custom_collate_for_mmgt)     
    return data_loader, dataset

def train(cfg):
    cfg_orig, exp_folder = load_model_config_exp(cfg['checkpoint_path'], cfg)
    cfg = merge_cfg(cfg, cfg_orig)
    cfg = merge_cfg(cfg, cfg['method_specs'])

    data_loader_name = "data_loader_train"
    valid_loader_name = "data_loader_valid"
    data_loader, dataset = create_train_dataset(batch_size=cfg['batch_size'], data_loader_name=data_loader_name, num_workers=0, 
                                                 shuffle=True, drop_last=False, **cfg)
    # data_loader, dataset = prepare_eval_dataset(cfg, 
    #                                             data_loader_name=data_loader_name, 
    #                                             drop_last=False, num_workers=0, 
    #                                             batch_size=cfg['batch_size'], dataset=dataset, 
    #                                             stats_mode='deterministic')
    # prepare_eval_dataset() switches off data augmentation during training and also changes how segments are sampled from sequences. If we want to keept it, we need to use create_train_dataset
    validset = set_updataset(cfg, valid_loader_name)
    valid_loader, validset = prepare_eval_dataset(cfg,
                                                  data_loader_name=valid_loader_name, 
                                                  drop_last=False, num_workers=0, 
                                                  batch_size=cfg['batch_size'], dataset=validset, 
                                                  stats_mode='deterministic')

    tb_logger = SummaryWriter(cfg['tb_dir'])
    logger = create_logger(os.path.join(cfg['log_dir'], 'log.txt'))
    display_exp_setting(logger, cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['device'] = device
    dct_m, idct_m_all = get_dct_matrix(cfg['t_pred'] + cfg['t_his'])
    cfg['dct_m_all'] = dct_m.float().to(device)
    cfg['idct_m_all'] = idct_m_all.float().to(device)
    
    model, diffusion = create_model_and_diffusion(cfg)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))
    
    trainer = Trainer(
        model = model,
        diffusion = diffusion,
        data_loader = data_loader,
        valid_loader = valid_loader,
        cfg = cfg,
        multimodal_dict = None,
        skeleton = dataset.skeleton,
        logger=logger,
        tb_logger=tb_logger
    )
    trainer.loop()

if __name__ == '__main__':
    main_hydra()