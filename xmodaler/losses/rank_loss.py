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
class RANKLoss(nn.Module):
    @configurable
    def __init__(self, loss_weight, eos_id):
        super(RANKLoss, self).__init__()
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
            logits = outputs_dict[kfg.U_LOGITS] #160x20x14
            targets = outputs_dict[kfg.U_TARGET_IDS] # already be encoded to bit representation in BitEmbed layer

            seq = outputs_dict[kfg.U_TOKENS_IDS] #160x20
            mask = (torch.cumsum((seq == self.eos_id), dim=-1) == 0) #超出句子长度的置为false即0
            mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).float() # 所有的mask往后移动一位

            # 把原本不属于句子的部分剔除
            mask = mask.unsqueeze(-1)  # 160x20x1
            logits = logits * mask  # 160x20x14
            targets = targets * mask
            mask_sum = torch.sum(mask.squeeze(-1), -1)
            # mask_sum = mask_sum

            pro = outputs_dict[kfg.G_LOGITS] #取预测单词的概率
            temp1 = torch.log_softmax(pro, -1) #LogSoftmax相对于Softmax，求导更快，还能解决上溢和下溢的问题
            temp1 = temp1 * mask #把原本不属于句子的部分剔除
            temp1[temp1==0]=-torch.inf #把原本不属于句子的部分置为-lnf，对剩下的部分进行topk排序
            (tmp1, tmp2) = torch.max(temp1, -1)#对于每个预测单词分别找出概率最大的作为预测单词，tmp1记录整个句子取当前每个预测单词的概率
            tmp3, tmp4 = torch.topk(tmp1, 5)#挑选tmp1中置信度topK个单词

            # tmp = torch.arange(mask.shape[0])[:, None]
            #按照topk选择的顺序，取出这些预测的单词
            logits = logits[torch.arange(mask.shape[0])[:, None], tmp4] #160x5x14
            # targets = targets[torch.arange(mask.shape[0])[:, None], tmp4]

            # 统计这批样本里面的rankloss,目的
            n = len(logits)
            preds = logits.view(n, -1) #800x14
            preds = torch.sum(preds, -1) #800x1
            preds = preds/mask_sum
            preds = preds.unsqueeze(0).repeat(n, 1) #800x800
            preds_t = preds.t() #800x800

            img_label = targets.view(n, -1) #160x280
            img_label = torch.mean(img_label, -1) #160x1
            img_label = img_label.unsqueeze(0).repeat(n, 1) #160x160
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


            ret.update({'Rank loss(U)': rank_loss*self.loss_weight})
        return ret


