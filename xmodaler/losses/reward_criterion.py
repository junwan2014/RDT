# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class RewardCriterion(nn.Module):
    @configurable
    def __init__(self, eos_id):
        super(RewardCriterion, self).__init__()
        self.eos_id = eos_id

    @classmethod
    def from_config(cls, cfg):
        return {
            'eos_id': cfg.SCORER.EOS_ID
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, outputs_dict):
        seq = outputs_dict[kfg.G_SENTS_IDS]#160x20
        logP = outputs_dict[kfg.G_LOGP]#160x20
        rewards = outputs_dict[kfg.REWARDS]#160

        mask = (torch.cumsum((seq == self.eos_id), dim=-1) == 0)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1) #160x20
        rewards = rewards.view(-1, 1).expand_as(logP)
        logP = torch.masked_select(logP, mask)
        rewards = torch.masked_select(rewards, mask)
        loss = torch.mean(-logP * rewards)

        return { 'RewardCriterion': loss }