"""Functions for searching face identity"""
import os
from glob import glob
from typing import List, Dict, Tuple, Any, Union
import cv2
import faiss
import h5py
import numpy as np
from tqdm import tqdm


def fast_search(src: np.ndarray, tar: np.ndarray,
                k_num: int, dim: int = 0, device="gpu") -> Tuple[np.ndarray, np.ndarray]:
    """Fast search based on cosine similarity by faiss model

    :param src: source matrix for initialization
    :param tar: target matrix for comparison
    :param dim: dimension of embedding vector
    :param k_num: number of nearest neighbor to search
    :param device: gpu or cpu for running fast search
    :return: return index and distance array based on the similarity
    """
    if dim == 0:
        dim = src.shape[-1]
    if device == "gpu" or device == "cuda" or k_num > 1024:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dim)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(src)
    distance, index_arr = index.search(tar, k_num)
    return distance, index_arr


def load_annotation(input_path: str) -> Dict[int, Any]:
    """Load existing annotation dict from npy file

    :param input_path: local folder for loading previous annotation information
    :return return the existing annotation from database
    """
    ann_dict = np.load(os.path.join(input_path, 'annotation.npy'), allow_pickle=True)
    ann_dict = ann_dict.item()

    return ann_dict


def load_embedding(input_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load existing embedding from h5 file and npy file

    :param input_path: local folder for loading previous annotation information
    :return return the existing annotation from database
    """
    if "image_id.npy" not in os.listdir(input_path):
        input_path = sorted(glob(input_path + "*"))[-1]

    src_id_list = np.load(os.path.join(input_path, 'image_id.npy'))
    with h5py.File(os.path.join(input_path, 'feature_arr.h5'), 'r') as hf_file:
        raw_src_emb_array = hf_file["feature_arr"][:]
    src_emb = norm_arr(raw_src_emb_array)

    return src_id_list, src_emb


def face_id(rest_keys: np.ndarray, rest_emb: np.ndarray, threshold: float, k_num: int = 5000, device: str = "default") \
        -> Tuple[Dict[str, Any], List]:
    """Face identification for the rest images
    :param rest_keys: keys of rest images
    :param rest_emb: feature embeddings of rest images
    :param threshold: threshold for cosine similarity
    :param k_num: number of images to search
    :param device: run on gpu or cpu
    
    :return: Dict of matching image keys, list of images with unique face identities
    """
    if rest_emb.size == 0:
        return {}, []
    distance, index_arr = fast_search(rest_emb, rest_emb, k_num=min(len(rest_keys), k_num),
                                      device=device)
    rest_rec = {}
    images_rec = []
    for i, dis_row in tqdm(enumerate(distance), total=len(rest_keys)):
        src_key = rest_keys[index_arr[i][0]]
        if src_key not in images_rec:
            for j in range(1, len(dis_row)):
                tar_key = rest_keys[index_arr[i][j]]
                if dis_row[j] > threshold and tar_key not in images_rec:
                    images_rec.append(src_key)
                    images_rec.append(tar_key)
                    if src_key not in rest_rec:
                        rest_rec[src_key] = [tar_key]
                    else:
                        rest_rec[src_key].append(tar_key)
    rest_keys_list = list(set(list(rest_keys)) - set(images_rec))
    return rest_rec, rest_keys_list


def assign_id(ann_dict: Dict[int, Any], rest_rec: Dict[str, Any],
              new_annotations: Dict[str, Any], rest_keys: list) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Assign new id for the rest images.
    :param ann_dict: save annotation into one dictionary
    :param rest_rec: annotation information from face identification
    :param new_annotations: final record of annotation
    :param rest_keys: list includes the rest images
    :return: dict of newly annotated faces, dict of all annotations
    """
    human_id = len(ann_dict.keys())
    for file_key, file_list in rest_rec.items():
        tar_keys = [file_key]
        tar_keys.extend(file_list)
        ann_dict[human_id] = tar_keys
        for file_id in tar_keys:
            new_annotations[file_id] = human_id
        human_id += 1
    for j, rest_key in enumerate(rest_keys):
        ann_dict[human_id + j] = [rest_key]
        new_annotations[rest_key] = human_id + j
    return new_annotations, ann_dict


def reverse_annotation(ann_dict: Dict[int, Any]) -> Dict[str, int]:
    """Reverse face identity annotation dictionary
    :param ann_dict: Dict that connects identity label with keys of images/faces 
    :return: Dict that connects image/face keys with its identity label
    """

    return {key: model_id for model_id in ann_dict.keys() for key in ann_dict[model_id]}


def recognition(ann_dict: Dict[int, Any], src_keys: np.ndarray, src_emb: np.ndarray, tar_keys: np.ndarray,
                tar_emb: np.ndarray, threshold: float, device: str = "cuda") \
        -> Tuple[Dict[int, Any], Dict[str, int], np.ndarray, np.ndarray]:
    """Face recognition for the input images
    :param ann_dict: annotation dictionary
    :param src_keys: keys of annotated images
    :param src_emb: embeddings of annotated images
    :param tar_keys: keys of new images
    :param tar_emb: embeddings of new images
    :param threshold: threshold for deciding if face belongs to same identity
    :param device: run on gpu or cpu
    :return: updated annotation dict, dict that assigns key of new faces to id labels, keys + embeddings of rest images
    """
    distance, index_arr = fast_search(src_emb, tar_emb, 1, device=device)
    new_annotations = {}
    rev_dict = reverse_annotation(ann_dict)

    rest_keys = []
    rest_emb = []

    for i, dis in tqdm(enumerate(distance[:, 0]), total=len(tar_keys)):
        if dis > threshold:
            ori_model_id = rev_dict[src_keys[index_arr[i][0]]]
            ann_dict[ori_model_id].append(tar_keys[i])
            new_annotations[tar_keys[i]] = ori_model_id
        else:
            rest_keys.append(tar_keys[i])
            rest_emb.append(tar_emb[i])
    return ann_dict, new_annotations, np.asarray(rest_keys), np.asarray(rest_emb)


def find_closest_ids(src_keys: Union[List[str], np.ndarray], src_emb: np.ndarray,
                     tar_keys: Union[List[str], np.ndarray], tar_emb: np.ndarray,
                     k_num: int = 5, device: str = "cuda") \
        -> Dict[str, List[Tuple[str, float]]]:
    """Face recognition for the input images

    :param src_keys: keys of annotated images
    :param src_emb: embeddings of annotated images
    :param tar_keys: keys of new images
    :param tar_emb: embeddings of new images
    :param k_num: how many clostest matches should be found
    :param device: run on gpu or cpu
    :return: updated annotation dict, dict that assigns key of new faces to id labels, keys + embeddings of rest images
    """

    dist, index_arr = fast_search(src_emb, tar_emb,  k_num=k_num, device=device)
    closest_ids = {}
    for i in range(dist.shape[0]):
        tar_key = tar_keys[i]
        closest_ids[tar_key] = []
        for j in range(dist.shape[1]):
            src_key = src_keys[index_arr[i][j]]
            dis = dist[i][j]
            closest_ids[tar_key].append((src_key, dis))

    return closest_ids


def identification(input_folder: str, emb_path="/home/face-recognition/annotation/", k_num=5,
                   device="gpu"):
    """Compare images with existing labels from database

    :param input_folder: input folder for images
    :param emb_path: local folder to load embedding vectors
    :param k_num: the number of faces for searching
    :param device: device to run 
    :return: list of closest face image ids
    """
    app = load_model()
    tar_keys = [image for image in os.listdir(input_folder) if os.path.splitext(image)[-1] in [".png", ".jpg", ".jpeg"]]
    target_images = [os.path.join(input_folder, image) for image in tar_keys]
    tar_emb = calculate_emb(app, target_images)
    src_keys, src_emb = load_embedding(emb_path)

    closest_ids = find_closest_ids(src_keys, src_emb, tar_keys, tar_emb, k_num=k_num, device=device)

    return closest_ids