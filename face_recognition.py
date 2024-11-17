"""Functions for face recognition"""
import os
from glob import glob
from typing import List, Dict, Tuple, Any, Union
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis


def load_model():
    """To load pretrained model for face recognition

    :return: face recognition model
    """
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                       allowed_modules=['detection', 'recognition', 'genderage'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def metric(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """To calculate metric for face recognition

    :param emb1: embedding vector for image1
    :param emb2: embedding vector for image2
    :return: return cosine metric
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def norm_arr(input_emb_array: np.ndarray) -> np.ndarray:
    """Normalization of the input embedding arrays

    :param input_emb_array: raw input embedding array 
    :return: normalized embedding array
    """
    return np.asarray([emb / np.linalg.norm(emb) for emb in input_emb_array])


def calculate_emb(app, target_images: List) -> np.ndarray:
    """To calculate feature embedding for face recognition tasks

    :param app: insight app service for calculating feature embedding vectors
    :param target_images: target images for calculating embedding vectors
    :return: return normalized feature embedding arrays
    """
    target_emb_list = []
    print(f"Found {len(target_images)} images")
    for img_name in tqdm(target_images):
        img = cv2.imread(img_name)
        faces = app.get(img)
        emb = faces[-1]['embedding']
        target_emb_list.append(emb / np.linalg.norm(emb))
    target_emb_array = np.asarray(target_emb_list)
    print("Finish calculation of embedding vectors")
    return target_emb_array


def comp_two_images(src, tar) -> float:
    """To calculate similarity for two face images

    :param src: source image path
    :param tar: target image path
    :return bool value for whether two images are the same person or not
    """
    src_img = cv2.imread(src)
    tar_img = cv2.imread(tar)
    app = init_app()
    emb = calculate_emb(app, [src_img, tar_img])

    return metric(emb[0], emb[1])




