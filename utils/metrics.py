from scipy.spatial.distance import pdist
import numpy as np
import torch

"""metrics"""


def compute_all_metrics(pred, gt):
    """
    calculate all metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]
        gt_multi: multi-modal ground truth, shape as [multi_modal, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde, mmade, mmfde
    """
    #if pred.shape[0] == 1:
    diversity = 0.0
    #dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    #diversity = dist_diverse.mean()

    #gt_multi = torch.from_numpy(gt_multi).to('cuda')
    #gt_multi_gt = torch.cat([gt_multi, gt], dim=0)

    #gt_multi_gt = gt_multi_gt[None, ...]
    gt = gt.reshape(gt.shape[0], gt.shape[1], -1)

    diff_multi = pred - gt
    dist = torch.linalg.norm(diff_multi, dim=2)
    # we can reuse 'dist' to optimize metrics calculation

    ade = dist.mean(dim=1)
    fde = dist[:, -1]
    ade = ade.mean()
    fde = fde.mean()

    return diversity, ade, fde
