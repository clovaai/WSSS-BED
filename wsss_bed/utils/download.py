"""
WSSS-BED (https://arxiv.org/abs/2404.00918)
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

import os
import tarfile
from zipfile import ZipFile

import wget

ACTIVATION_MAPS = {
    # VOC 2012
    "VOC2012_CAM": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_CAM.zip",
    "VOC2012_OAA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_OAA.zip",
    "VOC2012_MCIS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_MCIS.zip",
    "VOC2012_DRS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_DRS.zip",
    "VOC2012_EDAM": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EDAM.zip",
    # saliency-supervised methods: activation maps exist according to saliency maps
    "VOC2012_EPS_DeepUSPS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_DeepUSPS.zip",
    "VOC2012_EPS_MOVE": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_MOVE.zip",
    "VOC2012_EPS_DSS_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_DSS_COCO.zip",
    "VOC2012_EPS_DSS_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_DSS_COCO20.zip",
    "VOC2012_EPS_DSS_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_DSS_DUTS.zip",
    "VOC2012_EPS_DSS_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_DSS_MSRA_HKU.zip",
    "VOC2012_EPS_DSS_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_DSS_MSRA.zip",
    "VOC2012_EPS_PFAN_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PFAN_COCO.zip",
    "VOC2012_EPS_PFAN_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PFAN_COCO20.zip",
    "VOC2012_EPS_PFAN_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PFAN_DUTS.zip",
    "VOC2012_EPS_PFAN_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PFAN_MSRA_HKU.zip",
    "VOC2012_EPS_PFAN_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PFAN_MSRA.zip",
    "VOC2012_EPS_PoolNet_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PoolNet_COCO.zip",
    "VOC2012_EPS_PoolNet_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PoolNet_COCO20.zip",
    "VOC2012_EPS_PoolNet_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PoolNet_DUTS.zip",
    "VOC2012_EPS_PoolNet_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PoolNet_MSRA_HKU.zip",
    "VOC2012_EPS_PoolNet_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_PoolNet_MSRA.zip",
    "VOC2012_EPS_VST_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_VST_COCO.zip",
    "VOC2012_EPS_VST_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_VST_COCO20.zip",
    "VOC2012_EPS_VST_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_VST_DUTS.zip",
    "VOC2012_EPS_VST_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_VST_MSRA_HKU.zip",
    "VOC2012_EPS_VST_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_VST_MSRA.zip",
    "VOC2012_EPS_sal_DRS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_sal_DRS.zip",
    "VOC2012_EPS_sal_EDAM": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_sal_EDAM.zip",
    "VOC2012_EPS_sal_EPS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_sal_EPS.zip",
    "VOC2012_EPS_sal_L2G": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_sal_L2G.zip",
    "VOC2012_EPS_sal_OAA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_EPS_sal_OAA.zip",
    "VOC2012_L2G_DeepUSPS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_DeepUSPS.zip",
    "VOC2012_L2G_MOVE": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_MOVE.zip",
    "VOC2012_L2G_DSS_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_DSS_COCO.zip",
    "VOC2012_L2G_DSS_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_DSS_COCO20.zip",
    "VOC2012_L2G_DSS_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_DSS_DUTS.zip",
    "VOC2012_L2G_DSS_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_DSS_MSRA_HKU.zip",
    "VOC2012_L2G_DSS_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_DSS_MSRA.zip",
    "VOC2012_L2G_PFAN_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PFAN_COCO.zip",
    "VOC2012_L2G_PFAN_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PFAN_COCO20.zip",
    "VOC2012_L2G_PFAN_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PFAN_DUTS.zip",
    "VOC2012_L2G_PFAN_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PFAN_MSRA_HKU.zip",
    "VOC2012_L2G_PFAN_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PFAN_MSRA.zip",
    "VOC2012_L2G_PoolNet_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PoolNet_COCO.zip",
    "VOC2012_L2G_PoolNet_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PoolNet_COCO20.zip",
    "VOC2012_L2G_PoolNet_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PoolNet_DUTS.zip",
    "VOC2012_L2G_PoolNet_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PoolNet_MSRA_HKU.zip",
    "VOC2012_L2G_PoolNet_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_PoolNet_MSRA.zip",
    "VOC2012_L2G_VST_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_VST_COCO.zip",
    "VOC2012_L2G_VST_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_VST_COCO20.zip",
    "VOC2012_L2G_VST_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_VST_DUTS.zip",
    "VOC2012_L2G_VST_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_VST_MSRA_HKU.zip",
    "VOC2012_L2G_VST_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_VST_MSRA.zip",
    "VOC2012_L2G_sal_DRS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_sal_DRS.zip",
    "VOC2012_L2G_sal_EDAM": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_sal_EDAM.zip",
    "VOC2012_L2G_sal_EPS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_sal_EPS.zip",
    "VOC2012_L2G_sal_L2G": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_sal_L2G.zip",
    "VOC2012_L2G_sal_OAA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/VOC2012_L2G_sal_OAA.zip",
    # COCO 2014
    "COCO_CAM": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_CAM.zip",
    "COCO_DRS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_DRS.zip",
    # saliency-supervised methods: activation maps exist according to saliency maps
    "COCO_EPS_MOVE": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_MOVE.zip",
    "COCO_EPS_DSS_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_DSS_DUTS.zip",
    "COCO_EPS_DSS_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_DSS_MSRA_HKU.zip",
    "COCO_EPS_DSS_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_DSS_MSRA.zip",
    "COCO_EPS_PFAN_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_PFAN_DUTS.zip",
    "COCO_EPS_PFAN_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_PFAN_MSRA_HKU.zip",
    "COCO_EPS_PFAN_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_PFAN_MSRA.zip",
    "COCO_EPS_PoolNet_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_PoolNet_DUTS.zip",
    "COCO_EPS_PoolNet_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_PoolNet_MSRA_HKU.zip",
    "COCO_EPS_PoolNet_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_PoolNet_MSRA.zip",
    "COCO_EPS_VST_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_VST_DUTS.zip",
    "COCO_EPS_VST_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_VST_MSRA_HKU.zip",
    "COCO_EPS_VST_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_EPS_VST_MSRA.zip",
    "COCO_L2G_MOVE": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_MOVE.zip",
    "COCO_L2G_DSS_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_DSS_DUTS.zip",
    "COCO_L2G_DSS_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_DSS_MSRA_HKU.zip",
    "COCO_L2G_DSS_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_DSS_MSRA.zip",
    "COCO_L2G_PFAN_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_PFAN_DUTS.zip",
    "COCO_L2G_PFAN_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_PFAN_MSRA_HKU.zip",
    "COCO_L2G_PFAN_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_PFAN_MSRA.zip",
    "COCO_L2G_PoolNet_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_PoolNet_DUTS.zip",
    "COCO_L2G_PoolNet_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_PoolNet_MSRA_HKU.zip",
    "COCO_L2G_PoolNet_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_PoolNet_MSRA.zip",
    "COCO_L2G_VST_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_VST_DUTS.zip",
    "COCO_L2G_VST_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_VST_MSRA_HKU.zip",
    "COCO_L2G_VST_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/act_maps/COCO_L2G_VST_MSRA.zip",
}


SALIENCY_MAPS = {
    # VOC 2012
    # from unsupervised models
    "VOC2012_MOVE": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_MOVE.zip",
    "VOC2012_DeepUSPS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_DeepUSPS.zip",
    # saliency maps used in WSSS methods
    "VOC2012_sal_OAA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_sal_OAA.zip",
    "VOC2012_sal_DRS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_sal_DRS.zip",
    "VOC2012_sal_EDAM": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_sal_EDAM.zip",
    "VOC2012_sal_EPS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_sal_EPS.zip",
    "VOC2012_sal_L2G": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_sal_L2G.zip",
    # unified saliency maps: {DATASET}_{SOD_MODEL}_{SOD_DATASET}
    "VOC2012_DSS_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_DSS_COCO.zip",
    "VOC2012_DSS_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_DSS_COCO20.zip",
    "VOC2012_DSS_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_DSS_DUTS.zip",
    "VOC2012_DSS_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_DSS_MSRA_HKU.zip",
    "VOC2012_DSS_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_DSS_MSRA.zip",
    "VOC2012_PFAN_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PFAN_COCO.zip",
    "VOC2012_PFAN_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PFAN_COCO20.zip",
    "VOC2012_PFAN_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PFAN_DUTS.zip",
    "VOC2012_PFAN_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PFAN_MSRA_HKU.zip",
    "VOC2012_PFAN_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PFAN_MSRA.zip",
    "VOC2012_PoolNet_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PoolNet_COCO.zip",
    "VOC2012_PoolNet_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PoolNet_COCO20.zip",
    "VOC2012_PoolNet_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PoolNet_DUTS.zip",
    "VOC2012_PoolNet_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PoolNet_MSRA_HKU.zip",
    "VOC2012_PoolNet_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_PoolNet_MSRA.zip",
    "VOC2012_VST_COCO": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_VST_COCO.zip",
    "VOC2012_VST_COCO20": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_VST_COCO20.zip",
    "VOC2012_VST_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_VST_DUTS.zip",
    "VOC2012_VST_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_VST_MSRA_HKU.zip",
    "VOC2012_VST_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/VOC2012_VST_MSRA.zip",
    # COCO 2014
    # from unsupervised models
    "COCO2014_MOVE": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_MOVE.zip",
    "COCO2014_DeepUSPS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_DeepUSPS.zip",
    # saliency maps used in WSSS methods
    "COCO2014_sal_L2G": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_sal_L2G.zip",
    "COCO2014_sal_EPS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_sal_EPS.zip",
    # unified saliency maps: {DATASET}_{SOD_MODEL}_{SOD_DATASET}
    "COCO2014_DSS_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_DSS_DUTS.zip",
    "COCO2014_DSS_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_DSS_MSRA_HKU.zip",
    "COCO2014_DSS_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_DSS_MSRA.zip",
    "COCO2014_PFAN_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_PFAN_DUTS.zip",
    "COCO2014_PFAN_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_PFAN_MSRA_HKU.zip",
    "COCO2014_PFAN_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_PFAN_MSRA.zip",
    "COCO2014_PoolNet_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_PoolNet_DUTS.zip",
    "COCO2014_PoolNet_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_PoolNet_MSRA_HKU.zip",
    "COCO2014_PoolNet_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_PoolNet_MSRA.zip",
    "COCO2014_VST_DUTS": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_VST_DUTS.zip",
    "COCO2014_VST_MSRA_HKU": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_VST_MSRA_HKU.zip",
    "COCO2014_VST_MSRA": "https://github.com/clovaai/WSSS-BED/releases/download/sal_maps/COCO2014_VST_MSRA.zip",
}


def download_and_extract(url, dst, remove=True):
    wget.download(url, dst)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)
