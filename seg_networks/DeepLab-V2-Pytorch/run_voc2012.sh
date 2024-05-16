# python3 -m pip install addict
# python3 -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

GPU=0,1
CONFIG=configs/voc12_2gpu.yaml
DATA_ROOT="/mnt/tmp"
LABEL_DIR=SegmentationClassAug

SEEDS=("0" "1" "2")
for SEED in ${SEEDS[@]}; do
    LOG_DIR=Deeplabv2_voc12_${LABEL_DIR}_seed${SEED}
    
    CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
        --data_root ${DATA_ROOT} \
        --config_path ${CONFIG} \
        --label_dir ${LABEL_DIR} \
        --log_dir ${LOG_DIR} \
        --val_interval 100 \
        --random_seed ${SEED} \
        --amp
done
