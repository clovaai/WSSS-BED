import argparse
import importlib
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.multiprocessing import Process
from tqdm import tqdm

from metadata.dataset import load_img_id_list, load_img_label_list_from_npy
from network.resnet38d import Normalize
from util.download import download_and_extract
from util.imutils import HWC_to_CHW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--n_processes_per_gpu", nargs="*", type=int)
    parser.add_argument("--n_total_processes", default=1, type=int)
    parser.add_argument("--img_root", default="VOC2012", type=str)
    parser.add_argument("--saliency_root", default="VOC2012", type=str)
    parser.add_argument("--cam_npy", default=None, type=str)
    parser.add_argument("--thr", default=0.20, type=float)
    parser.add_argument("--sal_thr", default=None, type=float)
    parser.add_argument("--dataset", default="voc12", type=str)
    args = parser.parse_args()

    if args.dataset == "voc12":
        args.num_classes = 20
    elif args.dataset == "coco":
        args.num_classes = 80
    else:
        raise Exception("Error")

    # model information
    args.model_num_classes = args.num_classes

    # save path
    args.save_type = list()
    if args.cam_npy is not None:
        os.makedirs(args.cam_npy, exist_ok=True)
        args.save_type.append(args.cam_npy)

    # processors
    args.n_processes_per_gpu = [int(_) for _ in args.n_processes_per_gpu]
    args.n_total_processes = sum(args.n_processes_per_gpu)
    return args


def preprocess(image, scale_list, transform):
    img_size = image.size
    num_scales = len(scale_list)
    multi_scale_image_list = list()
    multi_scale_flipped_image_list = list()

    # insert multi-scale images
    for s in scale_list:
        target_size = (round(img_size[0] * s), round(img_size[1] * s))
        scaled_image = image.resize(target_size, resample=Image.CUBIC)
        multi_scale_image_list.append(scaled_image)

    # transform the multi-scaled image
    for i in range(num_scales):
        multi_scale_image_list[i] = transform(multi_scale_image_list[i])

    # augment the flipped image
    for i in range(num_scales):
        multi_scale_flipped_image_list.append(multi_scale_image_list[i])
        multi_scale_flipped_image_list.append(np.flip(multi_scale_image_list[i], -1).copy())

    return multi_scale_flipped_image_list


def predict_cam(model, image, label, gpu):
    original_image_size = np.asarray(image).shape[:2]
    # preprocess image
    multi_scale_flipped_image_list = preprocess(image, scales, transform)

    cam_list = list()
    model.eval()
    for i, image in enumerate(multi_scale_flipped_image_list):
        with torch.no_grad():
            image = torch.from_numpy(image).unsqueeze(0)
            image = image.cuda(gpu)
            cam = model.forward_cam(image)

            cam = F.interpolate(cam, original_image_size, mode="bilinear", align_corners=False)[0]
            cam = cam.cpu().numpy() * label.reshape(args.num_classes, 1, 1)

            if i % 2 == 1:
                cam = np.flip(cam, axis=-1)
            cam_list.append(cam)

    return cam_list


def infer_cam_mp(process_id, image_ids, label_list, cur_gpu):
    print("process {} starts...".format(os.getpid()))
    print(process_id, cur_gpu)
    print("GPU:", cur_gpu)
    print("{} images per process".format(len(image_ids)))

    model = getattr(importlib.import_module(args.network), "Net")(args.model_num_classes)
    model = model.cuda(cur_gpu)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    torch.no_grad()

    for i in tqdm(range(len(image_ids))):
        img_id = image_ids[i]
        label = label_list[i]

        # load image
        img_path = os.path.join(args.img_root, img_id + ".jpg")
        img = Image.open(img_path).convert("RGB")

        # infer cam_list
        cam_list = predict_cam(model, img, label, cur_gpu)

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        for j in range(args.num_classes):
            if label[j] > 1e-5:
                out_name = args.cam_npy + img_id + "_{}.png".format(j)
                cv2.imwrite(out_name, norm_cam[j] * 255.0)


def main_mp():
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_img_label_list_from_npy(image_ids, args.dataset)
    n_total_images = len(image_ids)
    assert len(image_ids) == len(label_list)

    saved_list = sorted([file[:-4] for file in os.listdir(args.save_type[0])])
    n_saved_images = len(saved_list)
    new_image_ids = list()
    new_label_list = list()
    for i, name in enumerate(image_ids):
        if True:  # name not in saved_list: # overwrite or not
            new_image_ids.append(name)
            new_label_list.append(label_list[i])
    image_ids = new_image_ids
    label_list = new_label_list

    n_total_processes = args.n_total_processes
    print("===========================")
    print("OVERALL INFORMATION")
    print("n_gpus:", n_gpus)
    print("n_processes_per_gpu", args.n_processes_per_gpu)
    print("n_total_processes:", n_total_processes)
    print("n_total_images:", n_total_images)
    print("n_saved_images:", n_saved_images)
    print("n_images_to_proceed", len(image_ids))
    print("===========================")

    sub_image_ids = list()
    sub_label_list = list()

    # split model and data
    split_size = len(image_ids) // n_total_processes
    for i in range(n_total_processes):
        # split image ids and labels
        if i == n_total_processes - 1:
            sub_image_ids.append(image_ids[split_size * i :])
            sub_label_list.append(label_list[split_size * i :])
        else:
            sub_image_ids.append(image_ids[split_size * i : split_size * (i + 1)])
            sub_label_list.append(label_list[split_size * i : split_size * (i + 1)])

    # multi-process
    gpu_list = list()
    for idx, num in enumerate(args.n_processes_per_gpu):
        gpu_list.extend([idx for i in range(num)])

    processes = list()
    for idx, process_id in enumerate(range(n_total_processes)):
        proc = Process(target=infer_cam_mp, args=(process_id, sub_image_ids[idx], sub_label_list[idx], gpu_list[idx]))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    args = parse_args()

    n_gpus = args.n_gpus
    scales = (0.5, 1.0, 1.5, 2.0)
    normalize = Normalize()
    transform = torchvision.transforms.Compose([np.asarray, normalize, HWC_to_CHW])

    if not os.path.exists(args.weights):
        # download pre-trained CAM weight
        download_and_extract("https://drive.google.com/uc?id=1d2E0RJASaEge6DyQ78ylPU-TzZI_qmkt", "classifier_cam.zip")

    main_mp()
