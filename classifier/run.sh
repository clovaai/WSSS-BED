# Default setting
GPU=0,1
DATASET=voc12

# NEED TO SET
DATASET_ROOT=YOUR_DATASET_ROOT
IMG_ROOT=${DATASET_ROOT}/JPEGImages
BACKBONE=resnet38_cls
SAVE_ROOT=ckpt
SESSION=voc12_cam
INFER_DATA=train_aug
TRAINED_WEIGHT=${SAVE_ROOT}/${SESSION}/checkpoint_cam.pth

CUDA_VISIBLE_DEVICES=${GPU} python3 infer.py \
    --dataset ${DATASET} \
    --infer_list metadata/${DATASET}/${INFER_DATA}.txt \
    --img_root ${IMG_ROOT} \
    --network network.${BACKBONE} \
    --weights ${TRAINED_WEIGHT} \
    --thr 0.0 \
    --sal_thr 0.2 \
    --n_gpus 2 \
    --n_processes_per_gpu 1 1 \
    --cam_npy ${SAVE_ROOT}/${SESSION}/activation_maps/
    
