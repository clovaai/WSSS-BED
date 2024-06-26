from .coco import COCO
from .cocostuff import CocoStuff10k, CocoStuff164k
from .voc import VOC, VOCAug


def get_dataset(name):
    return {
        "cocostuff10k": CocoStuff10k,
        "cocostuff164k": CocoStuff164k,
        "voc": VOC,
        "vocaug": VOCAug,
        "coco": COCO,
    }[name]
