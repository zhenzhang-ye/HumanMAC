import csv

import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(diffusion, data_loader, model, logger, cfg):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def get_prediction(data, pred, model_select):
        #traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        #traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        #traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        data = data.to(cfg['device'])
        obs = torch.cat([data, torch.zeros_like(pred).to(cfg['device'])], dim=1)
        obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(obs, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)

        traj_est = torch.matmul(cfg['idct_m_all'][:, :cfg['n_pre']], sampled_motion)
        # traj_est.shape (K, 125, 48)
        return traj_est

    stats_names = ['APD', 'ADE', 'FDE'] #'MMADE', 'MMFDE']
    stats_meter = {x: {y: AverageMeter() for y in ['HumanMAC']} for x in stats_names}
    gt = []
    pred = []
    count = 0
    for data in data_loader:
        pred_i_nd = get_prediction(data[0], data[1], model)
        pred.append(pred_i_nd)
        gt.append(data[1])
        count += 1
        if count == 2:
            break
    pred = torch.cat(pred, dim=0)
    # pred [50, 5187, 125, 48] in h36m
    pred = pred[:, cfg['t_his']:, :]
    gt = torch.cat(gt, dim=0).to(cfg['device'])
    apd, ade, fde = compute_all_metrics(pred, gt)
    stats_meter['APD']['HumanMAC'].update(apd)
    stats_meter['ADE']['HumanMAC'].update(ade)
    stats_meter['FDE']['HumanMAC'].update(fde)
    for stats in stats_names:
        str_stats = f'{stats}: ' + ' '.join(
            [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
        )
        logger.info(str_stats)

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg['result_dir'], 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['HumanMAC'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['HumanMAC'] = new_meter['HumanMAC']
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg['result_dir'])

    if os.path.exists(file_stat % cfg['result_dir']) is False:
        df1.to_csv(file_stat % cfg['result_dir'], index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg['result_dir'])
        df = pd.concat([df2, df1['HumanMAC']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg['result_dir'], index=False)
