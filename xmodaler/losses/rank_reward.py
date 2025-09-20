# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class RANKReward(nn.Module):
    @configurable
    def __init__(self, loss_weight, eos_id):
        super(RANKReward, self).__init__()
        self.eos_id = eos_id
        self.hard_thred = 1
        self.use_margin = False
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg):
        return {
            'eos_id': cfg.SCORER.EOS_ID,
            'loss_weight': cfg.LOSSES.RANKLOSS
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        ret  = {}
        if kfg.U_LOGITS in outputs_dict:
            targets =  torch.tensor(outputs_dict[kfg.BS_REWARDS][kfg.REWARDS]).cuda() #160
            logits = torch.tensor(outputs_dict[kfg.SA_REWARDS][kfg.REWARDS]).cuda() #160 already be encoded to bit representation in BitEmbed layer

            # 统计这批样本里面的rankloss,目的
            n = len(logits)

            preds = logits.unsqueeze(0).repeat(n, 1) #160x160
            preds_t = preds.t() #160x160

            img_label = targets.cuda().unsqueeze(0).repeat(n, 1) #160x160
            img_label_t = img_label.t() #160x160

            mask_time = outputs_dict[kfg.TIMES]
            mask_time = mask_time.unsqueeze(0).repeat(n, 1)
            mask_time_t = mask_time.t()  # 160x160
            masks_time = (torch.abs(mask_time - mask_time_t) < 0.12) & (
                        torch.abs(mask_time - mask_time_t) > 0)

            masks = torch.sign(img_label - img_label_t) #160x160
            masks = masks*masks_time
            masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (
                        torch.abs(img_label - img_label_t) > 0)
            masks_hard = masks_hard*masks_time

            if self.use_margin:
                rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
            else:
                rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
            rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)


            ret.update({'Rank reward(U)': rank_loss*self.loss_weight})
        return ret


