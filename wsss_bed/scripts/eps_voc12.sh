DATASET=voc
DATA_LIST=./data/voc2012/train_aug_cls.txt

SAL_DIR=saliency_maps/VOC2012_MOVE
ATM_DIR=activation_maps/VOC2012_EPS_MOVE
SAVE_DIR=pseudo_labels/VOC2012_EPS_MOVE

ATM_THRESH=0.2
IGNORE_LABEL=False
WO_SAL=True
WORKERS=8

python3 run.py --sal_dir ${SAL_DIR} --atm_dir ${ATM_DIR} --save_dir ${SAVE_DIR} --dataset ${DATASET} --data_list ${DATA_LIST} --atm_thresh ${ATM_THRESH} --ignore_label ${IGNORE_LABEL} --wo_sal ${WO_SAL} --workers ${WORKERS}