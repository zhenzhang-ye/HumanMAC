import torch
import os
from tqdm import tqdm
import numpy as np
import yaml
from utils.script import create_model_and_diffusion
from config import Config, update_config
from utils.script import dataset_split, sample_preprocessing
from utils.util import get_dct_matrix


# improt what you need

####################################
# TO DO
# Define following functions to prepare a baseline method for evaluation with eval.py
#################################################

def set_updataset(config, data_loader_name):
    """
    Impleent this function only if the baseline method requires a specific dataloader or a specific skeleton.
    Otherwise we use a dummy lambda that returns None
    """  
    return None


def load_model_config_exp(checkpoint_path, cfg):
    """
    Load the config file and the experiment folder from the checkpoint_path
    exp_folder: is a absolute or relative path
    cfg: dict of configuration
    """
    exp_folder = checkpoint_path
    method_cfg = Config(cfg['method_specs'], test=(cfg['method_specs']['model_mode'] != 'train'))
    method_cfg = update_config(vars(method_cfg), exp_folder)
    return method_cfg, exp_folder

def get_eval_out_folder(exp_folder, checkpoint_path, data_split, cfg):
    """
    Create folder where to store the evaluation results
    """
    stats_folder = cfg['cfg_dir']
    return stats_folder


def prepare_model(config, skeleton, silent=False, **kwargs):
    """
    This function must take as input a config_file and at least a second arg (skeleton) and return a laoded model on a device (cpu or cuda)
    model, device = prepare_model(config, skeleton,  **kwargs)
    """
    for i in range(torch.cuda.device_count()):
        if not silent:
            print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    dct_m, idct_m_all = get_dct_matrix(config['t_pred'] + config['t_his'])
    config['dct_m_all'] = dct_m.float().to(device)
    config['idct_m_all'] = idct_m_all.float().to(device)

    torch.set_grad_enabled(False)
    # Load models
    if not silent:
        # print('Loading Graph Model checkpoint: {} ...'.format(config['pretrained_GM_path']))
        print('Loading checkpoint: {} ...'.format(config['checkpoint_path']))
    model, diffusion = create_model_and_diffusion(config)
    model = model.to(device)
    ckpt = torch.load(config['ckpt_path'])
    model.load_state_dict(ckpt)
    model.eval()
    
    return [model, diffusion], device #, diffusion.num_timesteps toe eventually evaluate different diffusion timesteps



def get_prediction(obs, model, sample_num=50, config=None, **kwargs):
    """
    Generate sample_num predictions from model and input obs.
    return a dict or anything that will be used by process_evaluation_pair
    """
    obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
    obs = torch.concat([obs, torch.zeros(obs.shape[0], config['t_pred'], obs.shape[2]).to(obs.device)], dim=1)
    model, diffusion = model
    mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(
            obs, config, mode='metrics')
    sampled_motion = diffusion.sample_ddim(model,
                                        traj_dct,
                                        traj_dct_cond,
                                        mode_dict)

    traj_est = torch.matmul(config['idct_m_all'][:, :config['n_pre']], sampled_motion)
    traj_est = traj_est[:, None, ...]
    traj_est = traj_est.reshape(traj_est.shape[0], traj_est.shape[1], traj_est.shape[2], -1, 3)
    traj_est = traj_est[:, :, config['t_his']:, ...]
    return {'lat_pred': traj_est, 'pred': traj_est}


def process_evaluation_pair(dataset, target, pred_dict):
    """
    Process the target and the prediction and return them in the right format for metrics computation
    """
    ...
    pred, lat_pred = pred_dict['pred'], pred_dict['lat_pred'] # example
    batch_size, n_samples, seq_length, num_joints, features = pred.shape
    # batch_size, n_samples, n_diffusion_steps, seq_length, num_joints, features = pred.shape
    assert features == 3 and list(target.shape) == [batch_size, seq_length, num_joints, features]
    return target, pred, lat_pred
    
    