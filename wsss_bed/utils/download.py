"""
WSSS-BED (https://arxiv.org/abs/2404.00918)
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0
"""

import os
import tarfile
from zipfile import ZipFile

import gdown

ACTIVATION_MAPS = {
    # VOC 2012
    "VOC2012_CAM": "https://drive.google.com/uc?id=1ITTPbGVnkoB--8urAZLLPrsjGjZCPoee",
    "VOC2012_OAA": "https://drive.google.com/uc?id=1ZA5IBTTwSGRDG0VgUvRL4SmlQO8dexeA",
    "VOC2012_MCIS": "https://drive.google.com/uc?id=1Q2-VcnprxYX3GU1F1C4HxVAjTyIITBCo",
    "VOC2012_DRS": "https://drive.google.com/uc?id=1tsbBBhr9FKL-NZN0XDiVRfolQRkNSWqt",
    "VOC2012_EDAM": "https://drive.google.com/uc?id=15gjGRt1FD14Pr-xGY4b_FX9dqhvc4cYN",
    # saliency-supervised methods: activation maps exist according to saliency maps
    "VOC2012_EPS_DeepUSPS": "https://drive.google.com/uc?id=1ZR4ICgUKVksW4E4t4UPbjFKqMLfwEcTx",
    "VOC2012_EPS_MOVE": "https://drive.google.com/uc?id=1ULAjQwA17v9_eiBpHoUrKw5gMkKzj-A7",
    "VOC2012_EPS_DSS_COCO": "https://drive.google.com/uc?id=196Z0zQdWSv_l-myfenaELdDj6PfjSQBz",
    "VOC2012_EPS_DSS_COCO20": "https://drive.google.com/uc?id=1t5j1Q9ROFEnJZO4nljVLMdf0mdgD2Kav",
    "VOC2012_EPS_DSS_DUTS": "https://drive.google.com/uc?id=1ONlThrfVfzelzxkPdtQ0PfrBs69fpkgQ",
    "VOC2012_EPS_DSS_MSRA_HKU": "https://drive.google.com/uc?id=1ViC_-_ovqJpZnLk51PqxOQ56AY9X3mIq",
    "VOC2012_EPS_DSS_MSRA": "https://drive.google.com/uc?id=1qxkyMFcl82SC4df5-XcU0Bw7vzzwAMFm",
    "VOC2012_EPS_PFAN_COCO": "https://drive.google.com/uc?id=1V0Be-zIyH1JrTTGN4JWBnG5vAyoZqHeo",
    "VOC2012_EPS_PFAN_COCO20": "https://drive.google.com/uc?id=1lsAz0vojDGzp9Gj3_e1cVTyeahtec1tC",
    "VOC2012_EPS_PFAN_DUTS": "https://drive.google.com/uc?id=1RBxXlbLm9DC13VjuTrCvnr9qm5Eq-krs",
    "VOC2012_EPS_PFAN_MSRA_HKU": "https://drive.google.com/uc?id=10HsAqaChLc0Fw5zEL663SIXgWQKchvuC",
    "VOC2012_EPS_PFAN_MSRA": "https://drive.google.com/uc?id=17-gl4m7NMvxJF53GaUxSNcgJBLgZ-rm4",
    "VOC2012_EPS_PoolNet_COCO": "https://drive.google.com/uc?id=1WCkRIX-Kwtx6wnXWUV1I9_M_sH__Cp_9",
    "VOC2012_EPS_PoolNet_COCO20": "https://drive.google.com/uc?id=1eLYsfCfiiJgFF1X8WTQYgbuqI_8YSkd1",
    "VOC2012_EPS_PoolNet_DUTS": "https://drive.google.com/uc?id=1D9v_y5Bt4vkNwBKXbSCzf8vHyiveFQVk",
    "VOC2012_EPS_PoolNet_MSRA_HKU": "https://drive.google.com/uc?id=1DgtSA2avDWew-dAu7LcmjD0VHlfLxlyG",
    "VOC2012_EPS_PoolNet_MSRA": "https://drive.google.com/uc?id=1tckC7WeGXNPS_QTCWLPBIZd2Jj3JmEjI",
    "VOC2012_EPS_VST_COCO": "https://drive.google.com/uc?id=1tC7nYYAAXjRNk95JNxweT9fEYhyIuf4W",
    "VOC2012_EPS_VST_COCO20": "https://drive.google.com/uc?id=1XBpJqJQczDU7UX55-Ts4jbF_lg264m3h",
    "VOC2012_EPS_VST_DUTS": "https://drive.google.com/uc?id=19V7S82T7mcTLKElvRQV4QeMGIGQjjGFd",
    "VOC2012_EPS_VST_MSRA_HKU": "https://drive.google.com/uc?id=1DS5_lTYCTcqBHP2k8bbaeJEc8IhQoeTv",
    "VOC2012_EPS_VST_MSRA": "https://drive.google.com/uc?id=16iqN3_x2wTiDAd7NHlLoxLQycGeajy_u",
    "VOC2012_EPS_sal_DRS": "https://drive.google.com/uc?id=1FwiZnTPM--Aq_sjWLmnMORZakZVzxH5F",
    "VOC2012_EPS_sal_EDAM": "https://drive.google.com/uc?id=1LePy9IXHW0ZKmBfHbq25EIqwRPK8FqY0",
    "VOC2012_EPS_sal_EPS": "https://drive.google.com/uc?id=1OYy1QDE9Jn3Pafuw4MAkWswzHez00i39",
    "VOC2012_EPS_sal_L2G": "https://drive.google.com/uc?id=1mORNPY0rKWsmWAneUR6dJOyWRNLCEY_J",
    "VOC2012_EPS_sal_OAA": "https://drive.google.com/uc?id=1D6BljknV-wsBYVlBI8cW265gQGUjgzD8",
    "VOC2012_L2G_DeepUSPS": "https://drive.google.com/uc?id=1SbvdGDDWEU1D9AbnXPs_8hENeg6V9GXK",
    "VOC2012_L2G_MOVE": "https://drive.google.com/uc?id=10U3xp5fgVgOZC9kPZ6_hufX-Mmp5x322",
    "VOC2012_L2G_DSS_COCO": "https://drive.google.com/uc?id=1DJJl6C4WcjKn49noCwL1V1K0GChnnwj5",
    "VOC2012_L2G_DSS_COCO20": "https://drive.google.com/uc?id=1AJ_WUZF_yhA1Iu8GstQ1Yf_rnNw97co-",
    "VOC2012_L2G_DSS_DUTS": "https://drive.google.com/uc?id=1O6kx9C0wOjxv2xhmv0UcEpIM2jrmxLC0",
    "VOC2012_L2G_DSS_MSRA_HKU": "https://drive.google.com/uc?id=1_qTy_W2_vYQrAf9gJedqAxs75vMk9EAK",
    "VOC2012_L2G_DSS_MSRA": "https://drive.google.com/uc?id=1_mxVObZ6tlSNZS4X4fvkph26HqogMcnP",
    "VOC2012_L2G_PFAN_COCO": "https://drive.google.com/uc?id=1g17fu3tUonxowVP8jXt9QWs6Y_p7HYUL",
    "VOC2012_L2G_PFAN_COCO20": "https://drive.google.com/uc?id=15AknEcMPDKonZx5R_cNnfApJGfM8LOV3",
    "VOC2012_L2G_PFAN_DUTS": "https://drive.google.com/uc?id=1ngo06guFpQvHTdfATeJgSD8fJ7Xywk4D",
    "VOC2012_L2G_PFAN_MSRA_HKU": "https://drive.google.com/uc?id=1If4gLpYPjJFcnBXwoJGYhnQdLTbnJtpd",
    "VOC2012_L2G_PFAN_MSRA": "https://drive.google.com/uc?id=19De3f9ybS5qTGit7MfapS5p0hHFuwZRX",
    "VOC2012_L2G_PoolNet_COCO": "https://drive.google.com/uc?id=1p1wawDqrsd1KE90wuVxqE06G3ugAYy1a",
    "VOC2012_L2G_PoolNet_COCO20": "https://drive.google.com/uc?id=1rouq34ezeruM8xzfR7aRiDpsNp8RUcB1",
    "VOC2012_L2G_PoolNet_DUTS": "https://drive.google.com/uc?id=1-DeHghQcLTMfbi2uLI7x6K0zHjG-7bIy",
    "VOC2012_L2G_PoolNet_MSRA_HKU": "https://drive.google.com/uc?id=1xPfnRjO7sBC5VsJjFujtEg3zjJhXclOT",
    "VOC2012_L2G_PoolNet_MSRA": "https://drive.google.com/uc?id=1Yeu7hpuO3QTiGUrVzqFBLemuR5R5RNyR",
    "VOC2012_L2G_VST_COCO": "https://drive.google.com/uc?id=14Yuy6RyxbNMyrPLZC24y9UazHIwsz1Zl",
    "VOC2012_L2G_VST_COCO20": "https://drive.google.com/uc?id=16beaFcfh25N3myQqEbZcXtVKHHEWxhdg",
    "VOC2012_L2G_VST_DUTS": "https://drive.google.com/uc?id=1Au3HsAoznjLwGjyZcYQ4j-7ZMWJfMRPu",
    "VOC2012_L2G_VST_MSRA_HKU": "https://drive.google.com/uc?id=1uvvkfwO6mywzfrTM1BKmBJILxtTxT_Lp",
    "VOC2012_L2G_VST_MSRA": "https://drive.google.com/uc?id=1v0W-hA7c3ds9t3zySDvw1pUAEvN370S4",
    "VOC2012_L2G_sal_DRS": "https://drive.google.com/uc?id=10VPkPHUvIUU80EUCrnuYsfkK767dve0f",
    "VOC2012_L2G_sal_EDAM": "https://drive.google.com/uc?id=1I4tvARSmE-cbPdTyqI4w63IvacVYDWIJ",
    "VOC2012_L2G_sal_EPS": "https://drive.google.com/uc?id=15Zl46jALR9MNtqtxI_qvmH1ScbwGYvU4",
    "VOC2012_L2G_sal_L2G": "https://drive.google.com/uc?id=1r_N0xO1CsDVIwVvoBZUXxDOqCPri6Pnh",
    "VOC2012_L2G_sal_OAA": "https://drive.google.com/uc?id=1AubUHEmYS3BiaaWiNWdbPdrWcCoJkosg",
    # COCO 2014
    "COCO_CAM": "https://drive.google.com/uc?id=1t6DQc_UtD9AlxWjBqUhx5xb4_a-Yara2",
    "COCO_DRS": "https://drive.google.com/uc?id=15Myw-2gPrlvSo-prcnVhPgx7MFKNcbAo",
    # saliency-supervised methods: activation maps exist according to saliency maps
    "COCO_EPS_MOVE": "https://drive.google.com/uc?id=12OoJtnNy8aMsl-XM3sY0wWOg4mof48F1",
    "COCO_EPS_DSS_DUTS": "https://drive.google.com/uc?id=1e_kkp8oq-IQ_6HCi8w3l7qeo4k5kPcSm",
    "COCO_EPS_DSS_MSRA_HKU": "https://drive.google.com/uc?id=15kOZCVB2fEdU2EVqgjeAe8otDF73m6sI",
    "COCO_EPS_DSS_MSRA": "https://drive.google.com/uc?id=177Fg5l0uZZ3OGYipnQvD4Fmbldy6kp62",
    "COCO_EPS_PFAN_DUTS": "https://drive.google.com/uc?id=1YgqnhxWZ00oirfcqumJ1ggBoPnoXwZUy",
    "COCO_EPS_PFAN_MSRA_HKU": "https://drive.google.com/uc?id=1ZtgZieGuBstfP8YXghhJzYjM8TTDx2D6",
    "COCO_EPS_PFAN_MSRA": "https://drive.google.com/uc?id=1syD23iavwc-TpuJYtcAF6EAB7wZUv8T3",
    "COCO_EPS_PoolNet_DUTS": "https://drive.google.com/uc?id=16YkD5yRsFpWCBSxMwCHRc0wL0_17_b8y",
    "COCO_EPS_PoolNet_MSRA_HKU": "https://drive.google.com/uc?id=1pSvRDr-Sk7f_goB3URs6-vtwT5CBgo2_",
    "COCO_EPS_PoolNet_MSRA": "https://drive.google.com/uc?id=1v96dMTMvSCY-G3o6RcdO1o-7X2P5Eyiq",
    "COCO_EPS_VST_DUTS": "https://drive.google.com/uc?id=1cn00KFmKr5DXrkE2SNuLcMgcna6xZIAS",
    "COCO_EPS_VST_MSRA_HKU": "https://drive.google.com/uc?id=179edSzBcimqPHWcA_IoRiRp-7T27Q9T2",
    "COCO_EPS_VST_MSRA": "https://drive.google.com/uc?id=1RYBeRBqtxIcdytmVUAk_5bhxy_BofIwN",
    "COCO_L2G_MOVE": "https://drive.google.com/uc?id=1U3Z7WrWq7xnX28kXamvkmpQcGQJvVnP9",
    "COCO_L2G_DSS_DUTS": "https://drive.google.com/uc?id=1mlRGhUfunKK9dvtaCtBPt5QZaThKKD4K",
    "COCO_L2G_DSS_MSRA_HKU": "https://drive.google.com/uc?id=1MG8OBwBO41zgQScfpp-cqfaUi0Powdtv",
    "COCO_L2G_DSS_MSRA": "https://drive.google.com/uc?id=1qYYwvKmAe_-G_gTffG70wYB7e255Uvev",
    "COCO_L2G_PFAN_DUTS": "https://drive.google.com/uc?id=1LHk1M7mLKNkQ_E015aO0ZKtT09xgloI0",
    "COCO_L2G_PFAN_MSRA_HKU": "https://drive.google.com/uc?id=15YloSZZ31kX26Rp1GTfgyQKiz3NqNTj8",
    "COCO_L2G_PFAN_MSRA": "https://drive.google.com/uc?id=1YH6TFbztBBzIGR0lVgnV7wNsA13mJPHc",
    "COCO_L2G_PoolNet_DUTS": "https://drive.google.com/uc?id=1XS-AxsE3WadPU7VN0KKs3eEsMjdGnSaD",
    "COCO_L2G_PoolNet_MSRA_HKU": "https://drive.google.com/uc?id=1A7yaN4LzdjjQLPEbm6HuM-d8tyA985CI",
    "COCO_L2G_PoolNet_MSRA": "https://drive.google.com/uc?id=1f19d2f-UtLxowgRG0nDq5lGsIWWmlZph",
    "COCO_L2G_VST_DUTS": "https://drive.google.com/uc?id=1Ri8j0sLF0tkCfg4FXKCiGkVU5mWEP_v2",
    "COCO_L2G_VST_MSRA_HKU": "https://drive.google.com/uc?id=1--15hJhRmJTNb5NJPme8cmaoXMtuTYu7",
    "COCO_L2G_VST_MSRA": "https://drive.google.com/uc?id=1lp4FiWiq_L-pp13wwmLnsK45chuJYeiI",
}


SALIENCY_MAPS = {
    # VOC 2012
    # from unsupervised models
    "VOC2012_MOVE": "https://drive.google.com/uc?id=15gkzQJmDrM2N3rudYPcaK2X8L3XEMeHR",
    "VOC2012_DeepUSPS": "https://drive.google.com/uc?id=1-xnKMmfgZzNsyZdXTK8meps_woqzCb2q",
    # saliency maps used in WSSS methods
    "VOC2012_sal_OAA": "https://drive.google.com/uc?id=1sgjOkLqlKFbBZhFhJYnChTQ0WNh8JOCF",
    "VOC2012_sal_DRS": "https://drive.google.com/uc?id=1Z5FnVe3rQ7IoSG601W4vHq7RHjs9e4uD",
    "VOC2012_sal_EDAM": "https://drive.google.com/uc?id=1L4bIEQFEn6sAc62db0E9mxyv9pBOQELk",
    "VOC2012_sal_EPS": "https://drive.google.com/uc?id=1aGl12E-XNQ2RzYJIxwgfFCWwvNd8Z-q6",
    "VOC2012_sal_L2G": "https://drive.google.com/uc?id=1M-aQoevg27MkvC6hDuiwyj3p3uFpHruA",
    # unified saliency maps: {DATASET}_{SOD_MODEL}_{SOD_DATASET}
    "VOC2012_DSS_COCO": "https://drive.google.com/uc?id=14RfP99Nj1LvcjbXxWFcE-ged5Ifo_Bew",
    "VOC2012_DSS_COCO20": "https://drive.google.com/uc?id=1hDIgCwrdNlJr0ksAt1gCvOi6n7vdpNh8",
    "VOC2012_DSS_DUTS": "https://drive.google.com/uc?id=18q631gotaOZqA3Y4PKH5eNqoLFY9hHC8",
    "VOC2012_DSS_MSRA_HKU": "https://drive.google.com/uc?id=1L7Aj0wK4ZoWLTIfIiXnulmzKWnD-gwfw",
    "VOC2012_DSS_MSRA": "https://drive.google.com/uc?id=1a64_jMCPqrC7pSC9OLoNhL7XL2aB_ASr",
    "VOC2012_PFAN_COCO": "https://drive.google.com/uc?id=1Af6bbBVh5JTzuC8StPDHe-gETJjmBZ-Y",
    "VOC2012_PFAN_COCO20": "https://drive.google.com/uc?id=1w5GnAbaCYOYtgolcOLicaV3FsuLeXjMW",
    "VOC2012_PFAN_DUTS": "https://drive.google.com/uc?id=1umrcEi0B58886y_JXZuSJqKV73DTKUxo",
    "VOC2012_PFAN_MSRA_HKU": "https://drive.google.com/uc?id=13ZpVzJLTN2LDwvGqs5BnSkeAlaJ5-xJd",
    "VOC2012_PFAN_MSRA": "https://drive.google.com/uc?id=1CoK7-dewQ0vHBEPcWhUJs_O9gtWwVe7d",
    "VOC2012_PoolNet_COCO": "https://drive.google.com/uc?id=1gPkfRQ9KHxzpXcI63XsSS3igZhaYt1UZ",
    "VOC2012_PoolNet_COCO20": "https://drive.google.com/uc?id=1X4zJft3q46BaV7TVZ8Jogl-mKlDW096e",
    "VOC2012_PoolNet_DUTS": "https://drive.google.com/uc?id=17UqmFHtgKCML1Bnb7Xxahp6-qoSBjR0f",
    "VOC2012_PoolNet_MSRA_HKU": "https://drive.google.com/uc?id=1lN9tfYCfGztoeSzpDdzKgbP7nPAkq1nk",
    "VOC2012_PoolNet_MSRA": "https://drive.google.com/uc?id=1wrtC6X_66PtE417raJ7WWtEUxmdtpkA6",
    "VOC2012_VST_COCO": "https://drive.google.com/uc?id=1FbDLJJvgJeXyMjgQ8Qe7sI5irz_4mju6",
    "VOC2012_VST_COCO20": "https://drive.google.com/uc?id=1hfXB8gzWvW0RtuBvAP342gw_mI3YeVTS",
    "VOC2012_VST_DUTS": "https://drive.google.com/uc?id=1m4wjmAuvvkZKldHkg2aPUrGnuv5bs3VC",
    "VOC2012_VST_MSRA_HKU": "https://drive.google.com/uc?id=1ZotaqWa5s2B-VPGUyL4V5NKM31Px-DFa",
    "VOC2012_VST_MSRA": "https://drive.google.com/uc?id=1aNkilEi5Yl0UxQ3Tsa08PYxZMiYmmjq7",
    # COCO 2014
    # from unsupervised models
    "COCO2014_MOVE": "https://drive.google.com/uc?id=19QiVMCgfyVTf9ZEStWRmF3kxkry3DUi9",
    "COCO2014_DeepUSPS": "https://drive.google.com/uc?id=1O_Xhy2FzwL3bVQzKchn-fiwN8Lc2R9iQ",
    # saliency maps used in WSSS methods
    "COCO2014_sal_L2G": "https://drive.google.com/uc?id=10xi_b6JFwYsI9yX9b5j1kdy29oMmNkdv",
    "COCO2014_sal_EPS": "https://drive.google.com/uc?id=14NPp-tFYVNmpo4KNzIw42JLHyCs9MUZP",
    # unified saliency maps: {DATASET}_{SOD_MODEL}_{SOD_DATASET}
    "COCO2014_DSS_DUTS": "https://drive.google.com/uc?id=1NlbaW7Aga_KfziGgQXuDx5MU8HpG7BRw",
    "COCO2014_DSS_MSRA_HKU": "https://drive.google.com/uc?id=1LfjvgtXv0gfoTG01p1ZQQos-qlJf9GgM",
    "COCO2014_DSS_MSRA": "https://drive.google.com/uc?id=1RgUloG9l9OZ87bEyiwvw4J1ggqpHT3Um",
    "COCO2014_PFAN_DUTS": "https://drive.google.com/uc?id=1lu7UdfiTZzXUn-a6uSB79AC36CwkoBpd",
    "COCO2014_PFAN_MSRA_HKU": "https://drive.google.com/uc?id=1GmaCy1rvcm0eTo0WDfR0UUxQjNNTUXZs",
    "COCO2014_PFAN_MSRA": "https://drive.google.com/uc?id=1h2JjbImWMK6VI7Cfsx6mKG-4RkAsBeSA",
    "COCO2014_PoolNet_DUTS": "https://drive.google.com/uc?id=1YahZfjaDig-um1EpcK2Gd7b7WlgOEptU",
    "COCO2014_PoolNet_MSRA_HKU": "https://drive.google.com/uc?id=18fLSshZTXGcomrw-XdjVHhxQaXVvuBiT",
    "COCO2014_PoolNet_MSRA": "https://drive.google.com/uc?id=1z_JBePGEFCt0ZKYCMorJr4M_mKDyMIy4",
    "COCO2014_VST_DUTS": "https://drive.google.com/uc?id=1a95fErUFgRX6ws9SiXUkpU0jsShioEug",
    "COCO2014_VST_MSRA_HKU": "https://drive.google.com/uc?id=1Bj_fwgZagkFizCZtkr3jTddBIA6tIvd2",
    "COCO2014_VST_MSRA": "https://drive.google.com/uc?id=1zagSGFJ2ak6w__YWFH-i9qd79pnScA-n",
}


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)
