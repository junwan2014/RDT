# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from xmodaler.utils.registry import Registry
LOSSES_REGISTRY = Registry("LOSSES")
SCORER_REGISTRY = Registry("SCORER")
SCORER_REGISTRY.__doc__ = """
Registry for scorer
"""

def build_scorer(cfg):
    for name in cfg.SCORER.NAMES:
        if name in {'BaseScorer'}:
            score = SCORER_REGISTRY.get(name)(cfg)
    return score
