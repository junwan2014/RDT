# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import os
import copy
import pickle
import random
from tqdm import tqdm
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY
import warnings
import h5py

__all__ = ["MSCoCoDataset", "MSCoCoSampleByTxtDataset"]

@DATASETS_REGISTRY.register()
class MSCoCoDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        feats_folder: str,
        relation_file: str,
        gv_feat_file: str,
        attribute_file: str
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.max_seq_len = max_seq_len
        self.relation_file = relation_file
        self.gv_feat_file = gv_feat_file
        self.attribute_file = attribute_file
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_test.pkl")
        }

        if stage == 'test' and cfg.DATALOADER.INFERENCE_TRAIN == True:
            ann_file = str(os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_caption_anno_{}.pkl"))
        else:
            ann_file = ann_files[stage]

        ret = {
            "stage": stage,
            "anno_file": ann_file,
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "relation_file": cfg.DATALOADER.RELATION_FILE,
            "gv_feat_file": cfg.DATALOADER.GV_FEAT_FILE,
            "attribute_file": cfg.DATALOADER.ATTRIBUTE_FILE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN
        }
        return ret

    def _preprocess_datalist(self, datalist):
        return datalist

    def load_data(self, cfg):
        def _load_pkl_file(filepath):
            return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None

        if self.stage == 'test' and cfg.DATALOADER.INFERENCE_TRAIN == True:
            datalist = []
            for split in ['train', 'test']:
                anno_file = self.anno_file.format(split)
                tmp_datalist = pickle.load(open(anno_file, 'rb'), encoding='bytes')
                datalist.extend(tmp_datalist)
        else:
            datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')#读取训练标记数据mscoco_caption_anno_train.pkl

        if cfg.DEBUG:
            datalist = datalist[:100]
        datalist = self._preprocess_datalist(datalist)
        ext_data = {
            "relation": _load_pkl_file(self.relation_file),
            "attribute": _load_pkl_file(self.attribute_file),
            "gv_feat": _load_pkl_file(self.gv_feat_file)
        }
        for i in range(len(datalist)):
            image_id = int(datalist[i]['image_id'])
            for data_type in ext_data:
                if ext_data[data_type] is not None:
                    if str(image_id) in ext_data[data_type]:
                        datalist[i][data_type] = ext_data[data_type][str(image_id)]
                    elif image_id in ext_data[data_type]:
                        datalist[i][data_type] = ext_data[data_type][image_id]
        '''
        if len(self.relation_file) > 0:
            relation = pickle.load(open(self.relation_file, 'rb'), encoding='bytes')
            for i in range(len(datalist)):
                image_id = int(datalist[i]['image_id'])
                if image_id in relation:
                    datalist[i]['relation'] = relation[image_id]
        '''
        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = int(dataset_dict['image_id'])
        tmp = self.feats_folder

        try:
            if self.stage=='train':
                tmp = tmp + 'X152_grid_feats_coco_trainval.hdf5'
            else:
                tmp = tmp + 'X152_grid_feats_coco_trainval.hdf5'
            f = h5py.File(tmp, 'r')
            precomp_data = f['%d_grids' % image_id][()]

        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10, 2048)

        ret = {kfg.IDS: image_id, kfg.ATT_FEATS: precomp_data.astype(np.float32)}

        if 'relation' in dataset_dict:
            ret.update( { kfg.RELATION: dataset_dict['relation']} )
        if 'attribute' in dataset_dict:
            ret.update( { kfg.ATTRIBUTE: dataset_dict['attribute']} )
        if 'gv_feat' in dataset_dict:
            ret.update( { kfg.GLOBAL_FEATS: dataset_dict['gv_feat']} )
            
        if self.stage != 'train':
            g_tokens_type = np.zeros((self.max_seq_len,), dtype=np.int64)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
            dict_as_tensor(ret)
            return ret
        
        sent_num = len(dataset_dict['tokens_ids'])
        if sent_num >= self.seq_per_img:
            selects = random.sample(range(sent_num), self.seq_per_img) #随机取5个caption
        else:
            selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        tokens_ids = [ dataset_dict['tokens_ids'][i,:].astype(np.int64) for i in selects ] #第一个词向量从0开始
        target_ids = [ dataset_dict['target_ids'][i,:].astype(np.int64) for i in selects ] #把tokens_ids的每个词向量向前移动一个位置，句子最后一个位置为0，其余为-1，跟mask对应起来
        g_tokens_type = [ np.zeros((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ] #复制为0

        ret.update({
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
        })

        dict_as_tensor(ret)
        return ret


@DATASETS_REGISTRY.register()
class MSCoCoSampleByTxtDataset(MSCoCoDataset):
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        feats_folder: str,
        relation_file: str,
        gv_feat_file: str,
        attribute_file: str
    ):
        super(MSCoCoSampleByTxtDataset, self).__init__(
            stage,
            anno_file,
            seq_per_img, 
            max_feat_num,
            max_seq_len,
            feats_folder,
            relation_file,
            gv_feat_file,
            attribute_file
        )
        assert self.seq_per_img == 1

    def _preprocess_datalist(self, datalist):
        if self.stage == 'train':
            expand_datalist = []
            for data in tqdm(datalist, desc='Expand COCO Dataset'):
                for token_id, target_id in zip(data['tokens_ids'], data['target_ids']):
                    expand_datalist.append({
                        'image_id': data['image_id'],
                        'tokens_ids': np.expand_dims(token_id, axis=0),
                        'target_ids': np.expand_dims(target_id, axis=0)
                    })
            return expand_datalist
        else:
            return datalist