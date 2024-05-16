# WSSS-BED

For more meaningful and rigorous research for Weakly-Supervised Semantic Segmentation (WSSS).


### How to run?

~~~
DATASET=voc

SAL_DIR=saliency_maps/VOC2012_MOVE
ATM_DIR=activation_maps/CAM
SAVE_DIR=pseudo_labels/VOC2012_CAM_MOVE
ATM_THRESH=0.1
IGNORE_LABEL=True
WO_SAL=False
WORKERS=8

python3 run.py --sal_dir ${SAL_DIR} --atm_dir ${SAL_DIR --save_dir ${SAVE_DIR} --dataset ${DATASET} --atm_thresh ${ATM_THRESH} --ignore_label ${IGNORE_LABEL} --wo_sal ${WO_SAL} --workers ${WORKERS}
~~~

or

we provide the recommended setting for each WSSS method in `scripts/`.
~~~
bash scripts/cam_voc12.sh
bash scripts/eps_voc12.sh

bash scripts/cam_coco14.sh
bash scripts/l2g_coco14.sh
~~~



### Available WSSS Methods

We provide the [download link](https://github.com/qjadud1994/WSSS-BED/blob/feat/test/wsss_bed/utils/download.py#L7-L101) for the activation map from below WSSS method. 

- [CAM (CVPR'16)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
- [OAA (ICCV'19)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Integral_Object_Mining_via_Online_Attention_Accumulation_ICCV_2019_paper.pdf)
- [MCIS (ECCV'20)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470341.pdf)
- [DRS (AAAI'21)](https://ojs.aaai.org/index.php/AAAI/article/view/16269/16076)
- [EDAM (CVPR'21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Embedded_Discriminative_Attention_Mechanism_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2021_paper.pdf)
- [EPS (CVPR'21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.pdf)
- [L2G (CVPR'22)](https://openaccess.thecvf.com/content/CVPR2022/papers/Jiang_L2G_A_Simple_Local-to-Global_Knowledge_Transfer_Framework_for_Weakly_Supervised_CVPR_2022_paper.pdf)


### Available Saliency Maps

We provide the [download link](https://github.com/qjadud1994/WSSS-BED/blob/feat/test/wsss_bed/utils/download.py#L104-L157) for the below saliency map. 

- VOC2012
  - DSS_MSRA
  - DSS_MSRA_HKU
  - DSS_DUTS
  - DSS_COCO
  - DSS_COCO20
  - PFAN_MSRA
  - PFAN_MSRA_HKU
  - PFAN_DUTS
  - PFAN_COCO
  - PFAN_COCO20
  - PoolNet_MSRA
  - PoolNet_MSRA_HKU
  - PoolNet_DUTS
  - PoolNet_COCO
  - PoolNet_COCO20
  - VST_MSRA
  - VST_MSRA_HKU
  - VST_DUTS
  - VST_COCO
  - VST_COCO20
  - DeepUSPS
  - MOVE
  
- COCO2014
  - DSS_MSRA
  - DSS_MSRA_HKU
  - DSS_DUTS
  - PFAN_MSRA
  - PFAN_MSRA_HKU
  - PFAN_DUTS
  - PoolNet_MSRA
  - PoolNet_MSRA_HKU
  - PoolNet_DUTS
  - VST_MSRA
  - VST_MSRA_HKU
  - VST_DUTS
  - DeepUSPS
  - MOVE
  

### Source Contribution
Please make an issue first with your downloadable link of activation maps or saliency maps which is compatible with our framework.