# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import numpy as np
import pickle

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import SCORER_REGISTRY

__all__ = ['BaseScorer']

@SCORER_REGISTRY.register()
class BaseScorer(object):
    @configurable
    def __init__(
        self,
        *,
        types,
        scorers,
        weights,
        gt_path,
        eos_id
    ): 
       self.types = types
       self.scorers = scorers
       self.eos_id = eos_id
       self.weights = weights
       self.gts = pickle.load(open(gt_path, 'rb'), encoding='bytes')

    @classmethod
    def from_config(cls, cfg):
        scorers = []
        for name in cfg.SCORER.TYPES:
            scorers.append(SCORER_REGISTRY.get(name)(cfg))

        return {
            'scorers': scorers,
            'types': cfg.SCORER.TYPES,
            'weights': cfg.SCORER.WEIGHTS,
            'gt_path': cfg.SCORER.GT_PATH,
            'eos_id': cfg.SCORER.EOS_ID
        }

    def get_sents(self, sent):
        words = []
        for word in sent:
            if word == self.eos_id:
                words.append(self.eos_id)
                break
            words.append(word)
        return words

    def __call__(self, batched_inputs):
        ids = batched_inputs[kfg.IDS]
        gts = batched_inputs[kfg.G_TARGET_IDS]
        gts = gts.view(-1, 5, 20)
        gts_res = gts.cpu().tolist()
        gts_res_list = []
        for r in gts_res:
            gts_tmp = [self.get_sents(r_tmp) for r_tmp in r]
            gts_res_list.append(gts_tmp)
            gts_res_list.append(gts_tmp)
            gts_res_list.append(gts_tmp)
            gts_res_list.append(gts_tmp)
            gts_res_list.append(gts_tmp)

        res = batched_inputs[kfg.G_SENTS_IDS] #160x20
        res = res.cpu().tolist() #160个list
        #self.get_sents 把包含最后一个0的句子的有效部分取出来
        hypo = [self.get_sents(r) for r in res]

        rewards_info = {}
        rewards = np.zeros(len(ids))
        for i, scorer in enumerate(self.scorers): # gts_res_list：标记句子，每张图片对应5个caption, 160x5个list, hypo: 生成的句子，160个list
            score, scores = scorer.compute_score(gts_res_list, hypo)
            rewards += self.weights[i] * scores
            rewards_info[self.types[i]] = score
        rewards_info.update({ kfg.REWARDS: rewards })
        return rewards_info

