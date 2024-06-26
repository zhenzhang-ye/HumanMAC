import numpy as np
import os
import torch

from motpred_eval.motpred.dataset import FreeManDataset, H36MDataset 
from motpred_eval.motpred.dataset.base import MotionDataset
# from motpred_eval.motpred.dataset import create_skeleton

class HumanMacDataset(MotionDataset):
        
    # def recover_landmarks(data, rrr=True, fill_root=False):
    #     if rrr:
    #         assert data.shape[-2] == 17 or (data.shape[-2] == 16 and fill_root), "Root was dropped, so original landmarks can not be recovered"
    #         if data.shape[-2] == 16 and fill_root:
    #             # we fill with a 'zero' imaginary root
    #             size = list(data.shape[:-2]) + [1, data.shape[-1]] # (BatchSize, SegmentLength, NumPeople, 1, Dimensions)
    #             return np.concatenate((np.zeros(size), data), axis=-2) # same, plus 0 in the root position
    #         data[..., 1:, :] += data[..., :1, :]
    #     return data    
    
    
    # def preprocess_kpts(self, kpts):
    #     assert len(kpts.shape) == 3
    #     seq = kpts
    #     if self.use_vel:
    #         v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
    #         v = np.append(v, v[[-1]], axis=0)
    #     seq[:, 1:] -= seq[:, :1] # we make them root-relative (root-> joint at first position)
    #     if self.use_vel:
    #         seq = np.concatenate((seq, v), axis=1) # shape -> 17+1 (vel only from root joint)
    #     return seq

    def get_segment_from_dataset(self, idx):
        #obs, pred, extra = super(HumanMacDataset, self).get_segment_from_dataset(idx) 
        #data = np.concatenate([obs, pred], axis=-3)
        # data = self.preprocess_kpts(data) # we don't need to preprocess the keypoints, it is done by the skeleton class when calling tranform_to_input_space.
        #obs, pred = data[..., :obs.shape[-3], :, :], data[..., obs.shape[-3]:, :, :]
        # if self.if_load_mmgt:
        #     extra["mm_gt"] =  [self.preprocess_kpts(np.concatenate([obs, gt], axis=-3))[..., obs.shape[-3]:, :, :] for gt in extra["mm_gt"]] # if you are not using our skeleton: run preprocess_kpts() on the gt
        return super(HumanMacDataset, self).get_segment_from_dataset(idx) #obs, pred, extra
        #return obs, pred, extra

    '''
    def __getitem__(self, idx):
        obs, pred, extra = self.get_segment_from_dataset(idx)
        extra = self._get_mmgt_idx_(extra)
        obs, pred, extra = self.data_augmentation(obs, pred, extra)
        obs, pred, extra = self.tranform2inputspace(obs, pred, extra)
        return obs, pred, extra
    '''


class HumanMacFreeManDataset(HumanMacDataset, FreeManDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        
class HumanMacH36MDataset(HumanMacDataset, H36MDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "zhenzhang HumanMac"