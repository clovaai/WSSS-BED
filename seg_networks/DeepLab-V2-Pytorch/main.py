from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import time
import warnings

import joblib
import numpy as np
import yaml
from addict import Dict
from PIL import Image

from libs.datasets import get_dataset
from libs.models import *
from libs.utils import DenseCRF, PolynomialLR, scores
from libs.utils.download import download_and_extract
from libs.utils.stream_metrics import AverageMeter, StreamSegMetrics

warnings.filterwarnings(action="ignore")


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default=None, help="data root directory")
    parser.add_argument("--config_path", type=str, help="config file path")
    parser.add_argument("--label_dir", type=str, help="label directory")
    parser.add_argument("--log_dir", type=str, help="training log path")
    parser.add_argument("--cuda", type=bool, default=True, help="GPU")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--amp", action="store_true", default=False, help="enable half precision")
    parser.add_argument("--wo_crf", action="store_true", default=False, help="evaluate without DenseCRF")
    parser.add_argument("--val_interval", type=int, default=100, help="validation interval")

    return parser


def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def main():
    opts = get_argparser().parse_args()
    print(opts)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    """
    Training DeepLab by v2 protocol
    """
    # Configuration

    with open(opts.config_path) as f:
        CONFIG = Dict(yaml.safe_load(f))

    device = get_device(opts.cuda)
    torch.backends.cudnn.benchmark = True

    if opts.data_root is not None:
        CONFIG.DATASET.ROOT = opts.data_root

    # Dataset
    train_dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
        label_dir=opts.label_dir,
    )
    print(train_dataset)
    print()

    valid_dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
        # gt_path="SegmentationClassAug",
    )
    print(valid_dataset)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    # Model check
    print("Model:", CONFIG.MODEL.NAME)
    assert CONFIG.MODEL.NAME == "DeepLabV2_ResNet101_MSC", 'Currently support only "DeepLabV2_ResNet101_MSC"'

    # Model setup
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)

    if not os.path.exists(CONFIG.MODEL.INIT_MODEL):
        # download init weights
        download_and_extract("https://drive.google.com/uc?id=16fqLsWQswpATtOR7aQgm3yI6fhMfp5LK", "DLV2_init_weight.zip")

    print("    Init:", CONFIG.MODEL.INIT_MODEL, os.path.exists(CONFIG.MODEL.INIT_MODEL))
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL, map_location="cpu")

    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)

    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model = nn.DataParallel(model)
    model.to(device)

    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        # "models",
        # opts.log_dir,
        # CONFIG.MODEL.NAME.lower(),
        # CONFIG.DATASET.SPLIT.TRAIN,
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    def set_train(model):
        model.train()
        model.module.base.freeze_bn()

    metrics = StreamSegMetrics(CONFIG.DATASET.N_CLASSES)

    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    avg_loss = AverageMeter()
    avg_time = AverageMeter()

    set_train(model)
    best_score = 0
    end_time = time.time()

    for iteration in range(1, CONFIG.SOLVER.ITER_MAX + 1):
        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                _, images, labels, cls_labels = next(train_loader_iter)
            except:
                train_loader_iter = iter(train_loader)
                _, images, labels, cls_labels = next(train_loader_iter)
                avg_loss.reset()
                avg_time.reset()

            with torch.cuda.amp.autocast(enabled=opts.amp):
                # Propagate forward
                logits = model(images.to(device, non_blocking=True))

                # Loss
                iter_loss = 0
                for logit in logits:
                    # Resize labels for {100%, 75%, 50%, Max} logits
                    _, _, H, W = logit.shape
                    labels_ = resize_labels(labels, size=(H, W))

                    _loss = criterion(logit, labels_.to(device))

                    iter_loss += _loss

                # Propagate backward (just compute gradients wrt the loss)
                iter_loss /= CONFIG.SOLVER.ITER_SIZE

            scaler.scale(iter_loss).backward()
            loss += iter_loss.item()

        # Update weights with accumulated gradients
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step(epoch=iteration)

        avg_loss.update(loss)
        avg_time.update(time.time() - end_time)
        end_time = time.time()

        # TensorBoard
        if iteration % 20 == 0:
            print(
                " Itrs %d/%d, Loss=%6f, Time=%.2f , LR=%.8f"
                % (
                    iteration,
                    CONFIG.SOLVER.ITER_MAX,
                    avg_loss.avg,
                    avg_time.avg * 1000,
                    optimizer.param_groups[0]["lr"],
                )
            )

        # validation
        if iteration > (CONFIG.SOLVER.ITER_MAX * 0.6) and iteration % opts.val_interval == 0:
            print("... validation")
            model.eval()
            metrics.reset()
            with torch.no_grad():
                val_iter = 0
                for _, images, labels, _ in valid_loader:
                    val_iter += 1
                    if val_iter > 5000:
                        break
                    images = images.to(device, non_blocking=True)

                    # Forward propagation
                    logits = model(images)

                    # Pixel-wise labeling
                    _, H, W = labels.shape
                    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    targets = labels.cpu().numpy()
                    metrics.update(targets, preds)

            set_train(model)
            score = metrics.get_results()
            print(metrics.to_str(score))

            if score["Mean IoU"] > best_score:  # save best model
                best_score = score["Mean IoU"]
                torch.save(
                    model.module.state_dict(),
                    os.path.join(checkpoint_dir, f"checkpoint_best_s{opts.random_seed}.pth"),
                )
            end_time = time.time()

    # --------- validate -----------------------
    print("\n\n start validation...\n")
    best_ckpt = torch.load(os.path.join(checkpoint_dir, f"checkpoint_best_s{opts.random_seed}.pth"), map_location="cpu")

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)

    model.load_state_dict(best_ckpt, strict=True)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "val_features",
        # opts.log_dir,
        # CONFIG.MODEL.NAME.lower(),
        # CONFIG.DATASET.SPLIT.VAL,
        # "logit",
    )
    makedirs(logit_dir)

    metrics.reset()
    val_iter = 0
    for image_ids, images, gt_labels, cls_labels in valid_loader:
        val_iter += 1
        if val_iter % 5000 == 0:
            print("val Itrs %05d/%05d " % (val_iter, len(valid_loader)))
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)

        # Save on disk for CRF post-processing
        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            if not opts.wo_crf:
                np.save(filename, logit.detach().cpu().numpy())

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        metrics.update(gt_labels.detach().numpy(), labels.detach().cpu().numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = metrics.get_results()
    print("---------- best validation score without CRF ----------")
    print(metrics.to_str(score))
    print("----------------------------------------\n\n")

    if not opts.wo_crf:
        # CRF post-processor
        postprocessor = DenseCRF(
            iter_max=CONFIG.CRF.ITER_MAX,
            pos_xy_std=CONFIG.CRF.POS_XY_STD,
            pos_w=CONFIG.CRF.POS_W,
            bi_xy_std=CONFIG.CRF.BI_XY_STD,
            bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
            bi_w=CONFIG.CRF.BI_W,
        )

        pixel_mean = np.array((CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R))

        # Process per sample
        def process(i):
            image_id, image, gt_label, cls_labels = valid_dataset.__getitem__(i)

            filename = os.path.join(logit_dir, image_id + ".npy")
            logit = np.load(filename)

            _, H, W = image.shape
            logit = torch.FloatTensor(logit)[None, ...]
            logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
            prob = F.softmax(logit, dim=1)[0].numpy()

            image += pixel_mean[:, None, None]
            image = image.astype(np.uint8).transpose(1, 2, 0)
            prob = postprocessor(image, prob)
            label = np.argmax(prob, axis=0)

            return image_id, image, label, gt_label

        # CRF in multi-process
        n_jobs = 4  # multiprocessing.cpu_count() // 2
        results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
            [joblib.delayed(process)(i) for i in range(len(valid_dataset))]
        )

        image_ids, images, preds, gts = zip(*results)

        # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
        score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)
        print("---------- best validation score with CRF ----------")
        print(metrics.to_str(score))
        print("----------------------------------------\n\n")

    os.system(f"rm -rf {logit_dir}")


if __name__ == "__main__":
    main()
