import os
import json
import yaml
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import torch
from ignite.engine import Engine
from ignite.engine import Events, Engine, DeterministicEngine
from ignite.contrib.handlers import ProgressBar

from motpred_eval.motpred.utils.reproducibility import set_seed
from motpred_eval.motpred.metrics.utils import draw_table
from motpred_eval.motpred.metrics.multimodal import MetricStorer
from motpred_eval.motpred.utils.config import merge_cfg
from motpred_eval.motpred.eval_utils import get_stats_funcs, store_results_for_multiple_diffusion_steps, get_cmd_storer, prepare_eval_dataset

NDEBUG = False
USE_HYDRA = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_steps_to_evaluate(diffusion_stride, diffusion_steps):
    if diffusion_stride != -1:
        steps_to_evaluate = list(range(1, diffusion_steps+1, diffusion_stride))
        if diffusion_steps not in steps_to_evaluate:
            steps_to_evaluate += [diffusion_steps, ] # last step always evaluated, no matter what stride used
    else:
        steps_to_evaluate = [diffusion_steps, ] # only final result
    return steps_to_evaluate


def compute_metrics(dataset_split, store_folder, batch_size, multimodal_threshold=0.5, num_samples=50, 
                        prepare_model=None, get_prediction=None, process_evaluation_pair=None, set_updataset=None, 
                        silent=False, stats_mode="no_mm", metrics_at_cpu=False, **config):
                
    torch.set_default_dtype(torch.float64 if config["dtype"]== "float64" else torch.float32)

    data_loader_name = f"data_loader_{dataset_split}"
    if store_folder is not None:
        os.makedirs(store_folder, exist_ok=True)
    dataset = set_updataset(config, data_loader_name)    
    data_loader, dataset = prepare_eval_dataset(config, data_loader_name=data_loader_name, drop_last=False, num_workers=0, batch_size=batch_size, dataset=dataset, stats_mode=stats_mode)
    model, device = prepare_model(config, dataset.skeleton)
    stats_func = get_stats_funcs(stats_mode, **config)
    # for non diffusion methods, steps_to_evaluate = [1]
    steps_to_evaluate = [1] # get_steps_to_evaluate(diffusion_stride, diffusion_steps=num_timesteps) # for example [10], where last value is last diffusion timestep. 
    # steps_toenumerate = tqdm(range(len(steps_to_evaluate))) 
    print('Computing metrics at ', 'cpu.' if metrics_at_cpu else 'gpu.')
    
    def preprocess(engine: Engine):
        def mmgt_to_device(extradict):
            if 'mm_gt' in extradict:
                extradict['mm_gt'] = [mmgt.to(device) for mmgt in extradict['mm_gt']]
            return extradict
        engine.state.batch =  [t.to(device) if i<2 else mmgt_to_device(t) for i, t in enumerate(engine.state.batch)]
        
    def set_epoch_seed(engine: Engine):
        set_seed(0)

    def process_function(engine, batch):
        """ Process a batch and return elements necessary for metric computation

        Args:
            engine (ignite.engine.Engine): _description_
            batch (list of tensors and other structures): from dataloader

        Returns:
            dict: a dictionary with necessary entries for metric computation
                'pred': torch.Tensor of shape {batch, n_samples, n_diffusion_steps, pred_length, njoints, 3}. It is possible to set n_samples=1 (debug purposes or deterministic emthod)or n_diffusion_steps=1 (for non diffusion methods)
                'lat_pred': torch.Tensor of shape {batch, n_samples, n_diffusion_steps, latent_shape} with possible multiple latent shapes. necessary for latent_apd computation
                'target': torch.Tensor of shape {batch, pred_length, njoints, 3}
                'extra': dict or any structure holding for examle class information or similar
                'limbseq': list of tuples with the connections between joint idx. Necessary for limb_length metrics.
                
        """
        with torch.no_grad():
            data, target, extra = batch
            pred_dict = get_prediction(data, model, num_samples=num_samples, extra=extra, config=config) # [batch_size, n_samples, seq_length, num_joints, features]
            target, pred, lat_pred, mm_gt = process_evaluation_pair(dataset, target=target, pred_dict=pred_dict)
            if metrics_at_cpu:
                pred = pred.detach().cpu()
                target = target.detach().cpu()
                lat_pred = lat_pred.detach().cpu()
            return {'pred':pred, 'target':target, 'lat_pred':lat_pred, 
                    'extra':extra, 'limbseq': dataset.skeleton.limbseq, 'mm_gt': mm_gt}
    
    def extract_step(xdict, funct, step=0):
        assert step == 0, "Turned off. Only interesting to look intermediate diffusion outputs for generic diffusion methods. Output of method has to match. [batch, n_samples, n_diffusion_steps, pred_length, njoints, 3]"
        newdict = xdict.copy()
        # newdict['pred'] = newdict['pred'][:, :, step]
        # newdict['lat_pred'] = newdict['lat_pred'][:, :, step]
        return funct(**newdict)
        
    
    engine = Engine(process_function)
    engine.add_event_handler(Events.ITERATION_STARTED, preprocess)
    engine.add_event_handler(Events.EPOCH_STARTED, set_epoch_seed)
    pbar = ProgressBar()
    pbar.attach(engine)
    
    stats_metrics = {k+f'_step{step}': MetricStorer(output_transform=partial(extract_step, funct=funct, step=s) ,
                                                    return_op='max' if k=='LLErr_diffGT_max' else 'avg') for k, funct in stats_func.items() for s,step in enumerate(steps_to_evaluate)}
    # TO DO: something missing for MM metric classes.
        # mm_traj = mmgt_arr[counter: counter + target.shape[0]] if mmgt_arr is not None else None
        # values = stats_func[stats](target=target, pred=pred[:, :, step], gt_multi=mm_traj, lat_pred=lat_pred[:, :, step], limbseq=data_loader.dataset.skeleton.limbseq).cpu().numpy()
    for name, metric in stats_metrics.items():
        metric.attach(engine, name)
        
    if dataset_split=='test':
        config.pop('dataset')
        cmd_storer = get_cmd_storer(data_loader.dataset, **config)
        cmd_metrics = {cmd_name + f'_step{step}': cmdclass(output_transform=partial(extract_step, funct=out_funct, step=s)) 
                            for cmd_name, (cmdclass, out_funct) in cmd_storer.items() 
                            for s,step in enumerate(steps_to_evaluate)}
        for name, metric in cmd_metrics.items():
            metric.attach(engine, name)
    if NDEBUG:
        engine.run(data_loader, max_epochs=1, epoch_length=1) 
    else: 
        engine.run(data_loader)
    results = engine.state.metrics
    
    # store multiple steps of metrics
    if len(steps_to_evaluate)>1:
        store_results_for_multiple_diffusion_steps()
    
    # make latest step results the final one
    for name in list(results.keys()):
        if f'_step{steps_to_evaluate[-1]}' in name:
            results[name.replace(f'_step{steps_to_evaluate[-1]}', '')] = results[name]
    
    
    # ----------------------------- Printing results -----------------------------
    print('=' * 80)
    for table in draw_table(results):
        print(table)
    for stats in results:
        if stats.replace("traj", "").replace("pose", "") not in ['MPJPE', 'ADE', 'FDE', 'APD', 'MMADE', 'MMFDE']:
            if stats.replace("_max", "").replace("_mean", "").replace("_min", "") not in ['LLErr', 'LLErr_wrtGT', 'LLErr_diffGT', 'LLErr_GT']:
                print(f'Total {stats}: {results[stats]:.4f}')
    print('=' * 80)  
    # ----------------------------- Storing overall results -----------------------------
    
    # write results as json in plots folder
    ov_path = os.path.join(store_folder, f"results_{num_samples}.json")
    with open(ov_path, "w") as f:
        json.dump(str(results), f, indent=4)

    print(f"Overall results saved to {ov_path}")
    print('=' * 80)  
  
  
  
# model specific options go to hydra
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
OmegaConf.register_new_resolver("eval", eval)
@hydra.main(config_path="./motpred_interface/config", config_name="config")
def main_hydra(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # for backwards compatibility with the code
    for subconf in ['task', 'dataset']:
        if subconf in cfg:
            cfg = {**cfg, **cfg[subconf]}
            cfg.pop(subconf)
    main(**cfg)  

def main_argparse(args):
    assert 0, "Not implemented"
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    main(config)  
        
        
def main(checkpoint_path, method_name='mmugoon', dataset_split='valid', mode='stats', seed=0, stats_mode='motpred', **cfg):    
    

    assert mode in ['vis', 'gen', 'stats']
    assert mode == 'stats', "Only stats mode is implemented"

    """setup"""
    set_seed(seed)
    
    # build the config/checkpoint path
    if 'baseline' in method_name.lower():
        checkpoint_path = f"./output/baselines/{cfg['task_name']}/{method_name.lower().replace('baseline', '')}/{cfg['dataset_name']}"
    else:
        assert ".pt" in checkpoint_path, "Path should point to model save"
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Checkpoint not found in: %s" % checkpoint_path)
    
    
    # decide from which file to load the functions depending on which model we are evaluating
    if method_name == 'mmugoon':
        from motpred_eval.eval_prepare_model import prepare_model, get_prediction, process_evaluation_pair, load_model_config_exp, get_eval_out_folder
        set_updataset = lambda *args, **kwargs: None
    #elif 'baseline' in method_name.lower():
    #    from motpred_eval.prepare_algorithmic_baseline import prepare_model, get_prediction, process_evaluation_pair, load_model_config_exp, get_eval_out_folder
    #    set_updataset = lambda *args, **kwargs: None
    elif 'humanmac' in method_name.lower():
         from motpred_interface.motpred_prepare_humanmac import prepare_model, get_prediction, process_evaluation_pair, load_model_config_exp, get_eval_out_folder, set_updataset
    else:
        raise NotImplementedError()
    
    # load config
    cfg_orig, exp_folder = load_model_config_exp(checkpoint_path, cfg)
    # merge original experiment config with the current evaluation config
    cfg = merge_cfg(cfg, cfg_orig)
    cfg = merge_cfg(cfg, cfg['method_specs'])
    
    # set up evaluation functions
    # can also be done method dependent wise
    prepare_model = partial(prepare_model, **cfg) # check wheter we have later conflict because of method specs
    get_prediction = partial(get_prediction, **cfg)
    
    
    import warnings

    stats_folder = get_eval_out_folder(exp_folder, checkpoint_path, dataset_split, cfg)
    with open(os.path.join(stats_folder, 'eval_config.yaml'), 'w') as config_file:
        if USE_HYDRA:
            OmegaConf.save(cfg, config_file)
        else: 
            yaml.dump(cfg, config_file)
    print("Experiment data loaded from ", exp_folder)

    data_loader_name = f"data_loader_{dataset_split}"
    if 'segments_path' not in cfg[data_loader_name]:
        warnings.warn("We are not evaluating on segments")

    print(f"> Dataset: '{cfg['dataset_name']}'")
    print(f"> Exp name: '{exp_folder.split('/')[-1]}'")
    print(f"> Checkpoint: '{checkpoint_path.split('/')[-1]}'")
    print(f"> Prediction Horizon: '{cfg['pred_length']}'")

    if mode == 'vis' or mode == 'gen':
        store = 'gif' if mode == 'gen' else None # --> generate and store random generated sequences of a single batch.
        if store:
            print("Generating random sequences and storing them as 'gif'...")
        assert 0, "Not implemented"
        stats_folder = stats_folder + '_visuals'
    elif mode == 'stats':
        print(f"[WARNING] Remember: batch_size has an effect over the randomness of results. Keep batch_size fixed for comparisons, or implement several runs with different seeds to reduce stochasticity.")

        t0 = time.time()
        compute_metrics(dataset_split=dataset_split, stats_mode=stats_mode,
                        prepare_model=prepare_model, get_prediction=get_prediction, process_evaluation_pair=process_evaluation_pair, set_updataset=set_updataset,
                        store_folder=stats_folder, checkpoint_path=checkpoint_path, **cfg
                        )
        tim = int(time.time() - t0)
        print(f"[INFO] Evaluation took {tim // 60}min, {tim % 60}s.")

    else:
        raise NotImplementedError()


if __name__ == '__main__':
    if USE_HYDRA:
        main_hydra()
    else:
        parser = argparse.ArgumentParser()

        parser.add_argument('-c', '--checkpoint', required=True, help='path to checkpoint to load')
        parser.add_argument('-m', '--mode', default='stats', type=str, help='vis: visualize results\ngen: generate and store all visualizations for a single batch\nstats: launch numeric evaluation')
        parser.add_argument('-stats_mode', '--stats_mode', type=str, default="motpred")
        parser.add_argument('-b', '--batch_size', type=int, default=512)
        parser.add_argument('--multimodal_threshold', type=float, default=0.5)
        parser.add_argument('-cpu', '--cpu', action='store_true')
        parser.add_argument('-s', '--samples', type=int, default=-1)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('-d', '--data', default='valid')
        parser.add_argument('--pred_length', type=int, default=-1, help='evalaute for a different prediction length. Works only if segments are not given or if they are at least as long')
        parser.add_argument('--silent', action='store_true')
        parser.add_argument('--n_gpu', type=int, default=1)
        parser.add_argument('-sampler', '--sampler', default='ddim', help=f"options={list(SAMPLERS.keys())}")


        # Specs about model to be evaluated
        parser.add_argument('--diffusion_stride', type=int, default=-1)
        parser.add_argument('-e', '--ema', action='store_true')
        
        # args for visuals 
        # parser.add_argument('--ncols', type=int, default=0)
        # parser.add_argument('-t', '--type', default='3d') # 2d or 3d
        # parser.add_argument('-store_idx', '--store_idx', type=int, default=-1) # index of diffusion step to be stored
        args = parser.parse_args()
        main_argparse(args)