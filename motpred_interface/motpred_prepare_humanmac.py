import torch
import os
from tqdm import tqdm
import numpy as np
import yaml
from utils.script import create_model_and_diffusion
from config import Config, update_config
from utils.script import dataset_split, sample_preprocessing
from utils.util import get_dct_matrix
from motpred_eval.motpred.dataset import create_skeleton
from motpred_eval.motpred.utils.config import init_obj
from motpred_interface import motpred_dataset_humanmac as dataset_type


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
    skeleton = create_skeleton(**config)
    if 'HumanMac' not in config["dataset_type"]:
        config["dataset_type"] = "HumanMac" + config["dataset_type"]
    split = (
        data_loader_name.split("_")[-1]
        if "eval" not in data_loader_name
        else data_loader_name.split("_")[-2]
    )
    dataset = init_obj(
        config,
        "dataset_type",
        dataset_type,
        split=split,
        skeleton=skeleton,
        **(config[data_loader_name]),
    )
    return dataset


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



def get_prediction(obs, model, num_samples=50, extra=None, config=None, **kwargs):
    """
    Generate sample_num predictions from model and input obs.
    return a dict or anything that will be used by process_evaluation_pair
    """
    pred = torch.zeros(obs.shape[0], num_samples, config['t_pred'], obs.shape[2], 3).to(obs.device)
    obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
    obs = torch.concat([obs, torch.zeros(obs.shape[0], config['t_pred'], obs.shape[2]).to(obs.device)], dim=1)
    model, diffusion = model
    for i in range(num_samples):
        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(
                obs, config, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model,
                                            traj_dct,
                                            traj_dct_cond,
                                            mode_dict)

        traj_est = torch.matmul(config['idct_m_all'][:, :config['n_pre']], sampled_motion)
        traj_est = traj_est.reshape(traj_est.shape[0], traj_est.shape[1], -1, 3)
        traj_est = traj_est[:, config['t_his']:, ...]
        pred[:, i, ...] = traj_est
    mm_gt = extra['mm_gt'] if 'mm_gt' in extra else None
    return {"lat_pred": torch.zeros_like(pred), "pred": pred, 'mm_gt': mm_gt}


def process_evaluation_pair(dataset, target, pred_dict):
    """
    Process the target and the prediction and return them in the right format for metrics computation
    """
    ...
    pred, lat_pred, mm_gt, obs = pred_dict['pred'], pred_dict['lat_pred'], pred_dict['mm_gt'], pred_dict['obs']# example
    batch_size, n_samples, seq_length, num_joints, features = pred.shape
    target = dataset.skeleton.transform_to_metric_space(target)
    pred = dataset.skeleton.transform_to_metric_space(pred)
    obs = dataset.skeleton.transform_to_metric_space(obs)
    mm_gt = [dataset.skeleton.transform_to_metric_space(gt) for gt in mm_gt] if mm_gt is not None else None
    # batch_size, n_samples, n_diffusion_steps, seq_length, num_joints, features = pred.shape
    assert features == 3 and list(target.shape) == [batch_size, seq_length, num_joints, features]
    return target, pred, lat_pred, mm_gt, obs
    
    