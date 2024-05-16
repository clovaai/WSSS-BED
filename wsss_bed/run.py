"""
WSSS-BED (https://arxiv.org/abs/2404.00918)
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from torch.multiprocessing import Process
from tqdm import tqdm

from utils.download import ACTIVATION_MAPS, SALIENCY_MAPS, download_and_extract
from utils.palette import get_palette


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def get_parse():
    parser = argparse.ArgumentParser(description="WSSS-BED implementation")
    parser.add_argument("--sal_dir", type=str, default="./saliency_map/temp", help="target saliency map directory")
    parser.add_argument("--atm_dir", type=str, default="./CAM", help="target attention map directory")
    parser.add_argument("--save_dir", type=str, default="./pseudo_labels/temp", help="pseudo-label directory")

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, choices=["voc", "coco"], default="voc")

    parser.add_argument("--sal_thresh", type=float, default=0.2, help="threshold for saliency map")
    parser.add_argument("--atm_thresh", type=float, default=0.0, help="threshold for attention map")

    parser.add_argument("--data_list", type=str, default="./data/voc2012/train_aug_cls.txt")
    parser.add_argument("--ignore_label", type=str2bool, default=False, help="including ignore label")
    parser.add_argument("--wo_sal", type=str2bool, default=False, help="run without using saliency maps")

    return parser.parse_args()


def run(args, pid):
    for n, (img_name, valid_cls) in enumerate(zip(tqdm(img_name_list), label_list)):
        if n % args.workers != pid:
            continue

        save_path = os.path.join(args.save_dir, "%s.png" % img_name)
        if os.path.exists(save_path):
            continue

        """ load saliency map"""
        sal_path = os.path.join(args.sal_dir, img_name + ".png")
        sal = np.uint8(Image.open(sal_path).convert("L"))

        sal = np.float32(sal) / 255.0
        if args.sal_thresh > 0:
            sal = (sal > args.sal_thresh).astype(np.float32)  # binarization

        H, W = sal.shape

        """ load activation map"""
        pred = np.zeros((args.num_classes + 1, H, W), dtype=np.float32)

        if args.wo_sal:
            pred[0, :, :] = args.atm_thresh
        else:
            pred[0, :, :] = 1.0 - sal  # background cue

        for c in valid_cls:
            cam_path = os.path.join(args.atm_dir, f"{img_name}_{c}.png")
            pred[c + 1] = cv2.imread(cam_path, 0).astype(np.float32) / 255.0

        """ pseudo-label generation """
        if args.atm_thresh > 0 and not args.wo_sal:
            pred[1:] = np.where(pred[1:] < args.atm_thresh, 0, pred[1:])  # object cue

        pred = pred.argmax(0).astype(np.uint8)

        if args.ignore_label and not args.wo_sal:
            # pixels regarded as background but confidence saliency values
            pred[(sal > args.sal_thresh) & (pred == 0)] = 255

        pred_save = Image.fromarray(pred)
        pred_save.putpalette(palette)
        pred_save.save(save_path)


if __name__ == "__main__":
    args = get_parse()
    print("==================================================")
    print("  saliency map: ", args.sal_dir)
    print("  attention map: ", args.atm_dir)
    print("  pseudo label dir: ", args.save_dir)
    print("==================================================")

    if args.dataset == "voc":
        args.num_classes = 20
    elif args.dataset == "coco":
        args.num_classes = 80

    """ download sources """
    os.makedirs(os.path.dirname(args.sal_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.atm_dir), exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.sal_dir):
        sal_map_name = os.path.basename(args.sal_dir)
        download_and_extract(SALIENCY_MAPS[sal_map_name], f"{args.sal_dir}.zip")

    if not os.path.exists(args.atm_dir):
        atm_map_name = os.path.basename(args.atm_dir)
        download_and_extract(ACTIVATION_MAPS[atm_map_name], f"{args.atm_dir}.zip")

    """ get data list """
    img_name_list = []
    label_list = []

    with open(args.data_list, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            l = line.strip().split()
            img_name_list.append(l[0])
            label_list.append(list(map(int, l[1:])))

    palette = get_palette(args.dataset)

    processes = []
    for i in range(args.workers):
        proc = Process(target=run, args=(args, i))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
