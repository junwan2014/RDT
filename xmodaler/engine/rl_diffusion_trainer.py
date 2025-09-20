# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import time
import copy
import torch
import torch.nn.functional as F
import random
from .defaults import DefaultTrainer
from xmodaler.scorer import build_scorer
from xmodaler.config import kfg
from xmodaler.losses import build_rl_losses
import xmodaler.utils.comm as comm
from .build import ENGINE_REGISTRY

__all__ = ['RLDiffusionTrainer']

@ENGINE_REGISTRY.register()
class RLDiffusionTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(RLDiffusionTrainer, self).__init__(cfg)
        self.scorer = self.build_scorer(cfg)
        self.losses = build_rl_losses(cfg)

    @classmethod
    def build_scorer(cls, cfg):
        return build_scorer(cfg)

    def run_step(self):
        start = time.perf_counter()
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter//self.iters_per_epoch)

            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
        data_time = time.perf_counter() - start

        seq_per_img = data[0][kfg.SEQ_PER_SAMPLE].item() #每张图片句子的数量
        data = comm.unwrap_model(self.model).preprocess_batch(data) #对图片进行预处理

        outputs_dict = self.model(data)
        logprobs = F.log_softmax(outputs_dict[kfg.G_LOGITS], dim=-1) #160x20x10200

        # argmax to select baseline sents
        greedy_sents_ids = logprobs.argmax(-1)#160x20得到句子
        baseline_inputs = {
            kfg.IDS: data[kfg.IDS],
            kfg.G_TARGET_IDS: data[kfg.G_TARGET_IDS],
            kfg.G_SENTS_IDS: greedy_sents_ids
        }
        bs_rewards = self.scorer(baseline_inputs)

        # sample to select decoding sents
        probs = torch.exp(logprobs) #160x20x10200
        batch_size, seq_len = probs.shape[:2]
        sample_sents_ids = torch.multinomial(probs.view(batch_size*seq_len, -1), 1).view(batch_size, seq_len)
        # sample_sents_ids: 160x20，取概率最大的值对应的单词
        if kfg.RL_SAMPLE_FIX_TOKENS_IDS in data:
            # random replace one sentences
            fix_tokens_ids = data[kfg.RL_SAMPLE_FIX_TOKENS_IDS] #160x20
            img_num = int(batch_size / seq_per_img)
            replace_inds = [random.randint(0, seq_per_img-1) for _ in range(img_num)] #32个list，存放随机被替换的序号
            replace_mask = torch.zeros((img_num, seq_per_img)) #32x5，生成mask初始化为0
            replace_mask[range(img_num), replace_inds] = 1 # 随机被替换的句子的mask设置为1
            replace_mask = replace_mask.cuda().view(-1, 1).long() #160x1
            sample_sents_ids = replace_mask * fix_tokens_ids + (1 - replace_mask) * sample_sents_ids #得到新的句子
        #取对应的概率
        sample_sents_ids_logp = torch.gather(logprobs, index=sample_sents_ids.unsqueeze(-1), dim=-1).view(batch_size, seq_len)

        sample_inputs = {
            kfg.IDS: data[kfg.IDS],
            kfg.G_TARGET_IDS: data[kfg.G_TARGET_IDS],
            kfg.G_SENTS_IDS: sample_sents_ids,
        }
        sa_rewards = self.scorer(sample_inputs) #计算采样的句子和标记之间的scorer得分
        #将采样句子的score和贪心选择的score之间的误差作为损失，拟合整个网络。
        rewards = torch.from_numpy(sa_rewards[kfg.REWARDS] - bs_rewards[kfg.REWARDS]).float().cuda()
        outputs_dict.update({ 
            kfg.G_SENTS_IDS: sample_sents_ids,
            kfg.G_LOGP: sample_sents_ids_logp,
            kfg.REWARDS: rewards,
            kfg.BS_REWARDS: bs_rewards,
            kfg.SA_REWARDS: sa_rewards
        })

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        
        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]
        losses = sum(losses)

        self.optimizer.zero_grad()
        losses.backward()

        bs_rewards.pop(kfg.REWARDS)
        losses_dict.update(bs_rewards)

        # record pos/neg ratio
        pos_ratio = ((rewards > 0).sum() / rewards.shape[0]).item()
        neg_ratio = ((rewards < 0).sum() / rewards.shape[0]).item()
        losses_dict.update({
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio
        })

        self._write_metrics(losses_dict, data_time)
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)