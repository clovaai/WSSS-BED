# DeepLab-V2-Pytorch

The implementation of DeepLab-V2 is based on https://github.com/kazuto1011/deeplab-pytorch.

### VOC 2012

~~~
GPU=0,1
CONFIG=configs/voc12_2gpu.yaml
DATA_ROOT=YOUR_DATA_ROOT
LABEL_DIR=YOUR_PSEUDO_LABEL_DIR
LOG_DIR=Deeplabv2_voc12_${LABEL_DIR}_seed${SEED}
    
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    --data_root ${DATA_ROOT} \
    --config_path ${CONFIG} \
    --label_dir ${LABEL_DIR} \
    --log_dir ${LOG_DIR} \
    --val_interval 100 \
    --amp
~~~

or

~~~
bash run_voc2012.sh
~~~

### COCO 2014

~~~
GPU=0,1,2,3
CONFIG=configs/coco14_4gpu.yaml
DATA_ROOT=YOUR_DATA_ROOT
LABEL_DIR=YOUR_PSEUDO_LABEL_DIR
LOG_DIR=Deeplabv2_coco14_${LABEL_DIR}_seed${SEED}
    
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    --data_root ${DATA_ROOT} \
    --config_path ${CONFIG} \
    --label_dir ${LABEL_DIR} \
    --log_dir ${LOG_DIR} \
    --val_interval 100 \
    --amp \
    --wo_crf
~~~

or

~~~
bash run_cco2014.sh
~~~
