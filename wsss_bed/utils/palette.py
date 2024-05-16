"""
WSSS-BED (https://arxiv.org/abs/2404.00918)
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

import numpy as np


def get_palette(dataset):
    palette = []
    for i in range(256):
        palette.extend((i, i, i))

    if dataset == "voc":
        palette[: 3 * 21] = np.array(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ],
            dtype="uint8",
        ).flatten()
    else:
        palette[: 3 * 81] = np.array(
            [
                [0, 0, 0],
                [220, 20, 60],
                [119, 11, 32],
                [0, 0, 142],
                [0, 0, 230],
                [106, 0, 228],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 70],
                [0, 0, 192],
                [250, 170, 30],
                [100, 170, 30],
                [220, 220, 0],
                [175, 116, 175],
                [250, 0, 30],
                [165, 42, 42],
                [255, 77, 255],
                [0, 226, 252],
                [182, 182, 255],
                [0, 82, 0],
                [120, 166, 157],
                [110, 76, 0],
                [174, 57, 255],
                [199, 100, 0],
                [72, 0, 118],
                [255, 179, 240],
                [0, 125, 92],
                [209, 0, 151],
                [188, 208, 182],
                [0, 220, 176],
                [255, 99, 164],
                [92, 0, 73],
                [133, 129, 255],
                [78, 180, 255],
                [0, 228, 0],
                [174, 255, 243],
                [45, 89, 255],
                [134, 134, 103],
                [145, 148, 174],
                [255, 208, 186],
                [197, 226, 255],
                [171, 134, 1],
                [109, 63, 54],
                [207, 138, 255],
                [151, 0, 95],
                [9, 80, 61],
                [84, 105, 51],
                [74, 65, 105],
                [166, 196, 102],
                [208, 195, 210],
                [255, 109, 65],
                [0, 143, 149],
                [179, 0, 194],
                [209, 99, 106],
                [5, 121, 0],
                [227, 255, 205],
                [147, 186, 208],
                [153, 69, 1],
                [3, 95, 161],
                [163, 255, 0],
                [119, 0, 170],
                [0, 182, 199],
                [0, 165, 120],
                [183, 130, 88],
                [95, 32, 0],
                [130, 114, 135],
                [110, 129, 133],
                [166, 74, 118],
                [219, 142, 185],
                [79, 210, 114],
                [178, 90, 62],
                [65, 70, 15],
                [127, 167, 115],
                [59, 105, 106],
                [142, 108, 45],
                [196, 172, 0],
                [95, 54, 80],
                [128, 76, 255],
                [201, 57, 1],
                [246, 0, 122],
                [191, 162, 208],
            ],
            dtype="uint8",
        ).flatten()

    return palette
