This repo includes scripts for human face images processing.


# Face-reconstruction

face-reconstruction.py 3D human face reconstruction


# fast-face-recognition

The code is able to achieve face recognition and fast search among millions of personal identities. It is mainly based on two libraries [InsightFace](https://github.com/deepinsight/insightface#insightface-2d-and-3d-face-analysis-project) and [faiss](https://github.com/facebookresearch/faiss). 

## 1. Face recognition
[InsightFace](https://github.com/deepinsight/insightface#insightface-2d-and-3d-face-analysis-project) can generate predictions like face identity, age, and gender. More details can be found on their project page. 
Please follow https://github.com/deepinsight/insightface/tree/master/python-package to install InsightFace and download the pretrained model. You have to install onnxruntime-gpu manually to enable GPU inference, or install onnxruntime to use CPU only inference.


Here we use the generated embeddings for the following steps. We can compare embeddings from different images. If the similarity is lower than the threshold, these images are from the same person. You can set up different thresholds based on your requirement.

If you have a large set of face images, you can save all the embeddings for future search. My code can also create a face identity set based on your original images. 

## 2. Fast search
Fast search is achieved by using [faiss](https://github.com/facebookresearch/faiss). You can search one target face id among millions of face images or face identities. The code can be run by CPU or GPU.
Follow https://github.com/facebookresearch/faiss/blob/main/INSTALL.md to install faiss.