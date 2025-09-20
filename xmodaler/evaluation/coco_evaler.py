# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import tempfile
import json
from json import encoder
from xmodaler.config import kfg
from xmodaler.config import configurable
from xmodaler.utils import comm
from .build import EVALUATION_REGISTRY
from myevaluation import PTBTokenizer, compute_scores

sys.path.append(kfg.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

@EVALUATION_REGISTRY.register()
class COCOEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        if not os.path.exists(kfg.TEMP_DIR):
            os.mkdir(kfg.TEMP_DIR)

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir) and comm.is_main_process():
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

    def eval(self, results, epoch):
        if self.output_dir is not None: #测试指标计算
            json.dump(results, open(os.path.join(self.output_dir, str(epoch) + '.json'), "w"))

        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=kfg.TEMP_DIR)
        json.dump(results, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)

        # 修改之后的方法
        imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = cocoRes.imgToAnns[imgId]
        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        scores, _ = compute_scores(gts, res)

        #原方法
        # cocoEval = COCOEvalCap(self.coco, cocoRes)
        # cocoEval.evaluate()
        # os.remove(in_file.name)

        return scores