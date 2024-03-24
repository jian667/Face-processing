from typing import Tuple, List
import pandas as pd
import dlib
import numpy as np
import torch
import h5py
import cv2
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import trimesh


class PCAModel:
    def __init__(self, mean: np.ndarray, pc: np.ndarray, std: np.ndarray):
        """
        Initializes a PCAModel object.

        Parameters:
        - mean (np.ndarray): The mean vector of the PCA model.
        - pc (np.ndarray): The principal components matrix of the PCA model.
        - std (np.ndarray): The standard deviation vector of the PCA model.
        """
        self.mean = mean
        self.pc = pc
        self.std = std

def detect_landmark(image) -> np.ndarray:
    """
    Detect facial landmarks in an image using dlib.

    Parameters:
    - image: The input image.

    Returns:
    - np.ndarray: Numpy array containing (x, y)-coordinates of facial landmarks.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    dets = detector(image, 1)
    shape = predictor(image, dets[0])
    return shape_to_np(shape)

def shape_to_np(shape, dtype: str = "int") -> np.ndarray:
    """
    Convert dlib shape object to a numpy array of coordinates.

    Parameters:
    - shape: The dlib shape object.
    - dtype (str): Data type for the resulting numpy array.

    Returns:
    - np.ndarray: Numpy array containing (x, y)-coordinates of facial landmarks.
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def load_bfm_model(file_path: str, n_id: int, n_exp: int) \
    -> Tuple[PCAModel, PCAModel, PCAModel, torch.Tensor]:
    """
    Load the Basel Face Model (BFM) from an HDF5 file.

    Parameters:
    - file_path (str): The file path to the BFM HDF5 file.
    - n_id (int): The number of identity components to retain.
    - n_exp (int): The number of expression components to retain.

    Returns:
    - Tuple[PCAModel, PCAModel, PCAModel, torch.Tensor]: Tuple containing shape, color, 
    expression PCA models, and triangles.
    """
    bfm = h5py.File(file_path, 'r')

    shape_pca = load_pca_model(bfm, 'shape', n_id)
    tex_pca = load_pca_model(bfm, 'color', n_id)
    expr_pca = load_pca_model(bfm, 'expression', n_exp)

    triangles = torch.from_numpy(np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T)

    return shape_pca, tex_pca, expr_pca, triangles

def load_pca_model(bfm_group, model_name: str, n_id: int) -> PCAModel:
    """
    Load a PCA model from the Basel Face Model (BFM) group.

    Parameters:
    - bfm_group: The HDF5 group containing the PCA model data.
    - model_name (str): The name of the model (e.g., 'shape', 'color', 'expression').
    - n_id (int): The number of principal components to retain.

    Returns:
    - PCAModel: The loaded PCA model.
    """
    mean = torch.from_numpy(np.asarray(bfm_group[f'{model_name}/model/mean'], dtype=np.float32))
    pc = torch.from_numpy(np.asarray(bfm_group[f'{model_name}/model/pcaBasis'], dtype=np.float32))
    std = torch.sqrt(torch.from_numpy(np.asarray(bfm_group[f'{model_name}/model/pcaVariance'], \
                                                 dtype=np.float32)))
    pc = pc[:, :n_id]
    std = std[:n_id]
    return PCAModel(mean, pc, std)
# (pca.mean + torch.sum(coeff[0] * pca.std.view(1, -1) * pca.pc, dim=1)).view(1, -1, 3)

def rotate_euler(x, rot):
    """
    Rotate a 3D point cloud using Euler angles.

    Parameters:
    - x: The input 3D point cloud (torch.Tensor).
    - rot: Euler angles for rotation in the format [yaw, pitch, roll] (torch.Tensor).

    Returns:
    - torch.Tensor: The rotated 3D point cloud.
    """
    yaw, pitch, roll = rot[:, 0], rot[:, 1], rot[:, 2]

    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)

    yaw_c = torch.cos(yaw)
    pitch_c = torch.cos(pitch)
    roll_c = torch.cos(roll)

    yaw_s = torch.sin(yaw)
    pitch_s = torch.sin(pitch)
    roll_s = torch.sin(roll)

    # Rotation matrices for each axis
    roll = torch.stack([ones, zeros, zeros,
                        zeros, roll_c, -roll_s,
                        zeros, roll_s, roll_c], dim=1).view(-1, 3, 3)

    pitch = torch.stack([pitch_c, zeros, pitch_s,
                         zeros, ones, zeros,
                         -pitch_s, zeros, pitch_c], dim=1).view(-1, 3, 3)

    yaw = torch.stack([yaw_c, -yaw_s, zeros,
                       yaw_s, yaw_c, zeros,
                       zeros, zeros, ones], dim=1).view(-1, 3, 3)

    # Combine rotations
    rotation = torch.matmul(yaw, torch.matmul(pitch, roll))

    # Rotate the input point cloud
    return torch.matmul(x, rotation.permute(0, 2, 1))

def project_opengl(x, w, h, fovy=0.5):
    """
    Perform OpenGL-style perspective projection on a 3D point cloud.

    Parameters:
    - x (torch.Tensor): 3D point cloud with shape (batch, n_points, 3).
    - w (float): Width of the viewport.
    - h (float): Height of the viewport.
    - fovy (float): Vertical field of view angle in radians.

    Returns:
    - torch.Tensor: Projected 2D point cloud with shape (batch, n_points, 2).
    """
    w = torch.tensor(w, dtype=torch.float32)
    h = torch.tensor(h, dtype=torch.float32)

    batch, n_points, _ = x.shape
    cx = w / 2
    cy = h / 2
    fovy = torch.tensor(fovy, dtype=torch.float32)

    aspect = w / h
    t = torch.tan(fovy / 2)
    r = aspect * t

    f = 2000.
    n = 300.

    depth_range = f - n

    # Converts point cloud to NDC space
    # Projection matrix after which x, y, z \in [-1, 1]
    b1 = torch.tensor([1 / r, 0, 0, 0]).view(1,4)
    b2 = torch.tensor([0, 1 / t, 0, 0]).view(1,4)
    b3 = torch.tensor([0, 0, -(f + n) / depth_range, -2 * f * n / depth_range]).view(1,4)
    b4 = torch.tensor([0, 0, -1, 1]).view(1,4)
    projection_mat = torch.cat((b1, b2, b3, b4))

    # Converts point cloud to raster space
    # http://glasnost.itcarlow.ie/~powerk/GeneralGraphicsNotes/projection/viewport_transformation.html

    a1 = torch.tensor([cx, 0, 0, cx]).view(1,4)
    a2 = torch.tensor([0, -cy, 0, cy]).view(1,4)
    a3 = torch.tensor([0, 0, 0.5, 0.5]).view(1,4)
    a4 = torch.tensor([0, 0, 0, 1]).view(1,4)
    viewport_mat = torch.cat((a1, a2, a3, a4), 0)
    P = torch.matmul(viewport_mat, projection_mat).unsqueeze(0).expand(batch, viewport_mat.size(0), viewport_mat.size(1))
    homogeneous = torch.cat([x, torch.ones((batch, n_points, 1), dtype=torch.float32)], dim=2)
    projection = torch.matmul(homogeneous, P.permute(0, 2, 1))

    projection = projection.permute(2, 0, 1)
    w = projection[3:4]
    w = torch.maximum(torch.abs(w), torch.tensor(1e-6, dtype=torch.float32)) * torch.sign(w)
    projection = projection[:3] / w
    projection = projection.permute(1, 2, 0)
    return projection


def matplotlib_imshow(image, one_channel=False):
    """show image by matplotlib
    Args:
        image (_type_): input image
        one_channel (bool, optional): color channel of input image
    """
    if one_channel:
        plt.imshow(image, cmap="Greys")
    else:
        plt.imshow(image)

class LinearRegressionModel:
    """
    Define a simple linear regression model with two parameters (alpha and beta)
    """
    def __init__(self, n_id, n_exp):
        self.alpha = torch.zeros([1, n_id], requires_grad=True)
        self.delta = torch.zeros([1, n_exp], requires_grad=True)
        self.rotation = torch.zeros([1, 3], requires_grad=True)
        self.t = torch.tensor([0, 0, -400], dtype=torch.float32, requires_grad=True)

    def forward(self, pca1, pca2, img):
        p1 = (pca1.mean + torch.sum(self.alpha * pca1.std.view(1, -1) * pca1.pc, dim=1)).view(1, -1, 3)
        p2 = (pca2.mean + torch.sum(self.delta * pca2.std.view(1, -1) * pca2.pc, dim=1)).view(1, -1, 3)
        p0 = p1 + p2
        p = rotate_euler(p0, self.rotation) + self.t
        p = project_opengl(p, img.shape[1], img.shape[0])
        return p, p0

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function
    Args:
        y_true (_type_): ground truth
        y_pred (_type_): prediction

    Returns:
        _type_: mean squared loss
    """
    return torch.mean((y_true - y_pred)**2)

def landmark_loss(predict_lm, gt_lm, weight=None):
    """
    weighted mse loss
    Parameters:
        predict_lm    --torch.tensor (B, 68, 2)
        gt_lm         --torch.tensor (B, 68, 2)
        weight        --numpy.array (1, 68)
    """
    if not weight:
        weight = np.ones([68])
        weight[28:31] = 20
        weight[-8:] = 20
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(predict_lm.device)
    lm_loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    lm_loss = torch.sum(lm_loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return lm_loss

def reg_loss(model, opt=None):
    """
    l2 norm without the sqrt, from yu's implementation (mse)
    tf.nn.l2_loss https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    Parameters:
        coeffs_dict     -- a  dict of torch.tensors , keys: id, exp, tex, angle, gamma, trans

    """
    # coefficient regularization to ensure plausible 3d faces
    if opt:
        w_id, w_exp, w_tex = opt.w_id, opt.w_exp, opt.w_tex
    else:
        w_id, w_exp, w_tex = 1, 1, 1
    # creg_loss = w_id * torch.sum(coeffs_dict['id'] ** 2) +  \
    #        w_exp * torch.sum(coeffs_dict['exp'] ** 2) + \
    #        w_tex * torch.sum(coeffs_dict['tex'] ** 2)
    creg_loss = w_id * torch.sum(model.alpha ** 2) +  \
           w_exp * torch.sum(model.delta ** 2)
    # creg_loss = creg_loss / coeffs_dict['id'].shape[0]
    # gamma regularization to ensure a nearly-monochromatic light
    # gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
    # gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    # gamma_loss = torch.mean((gamma - gamma_mean) ** 2)
    return creg_loss
if __name__ == "__main__":
    torch.manual_seed(42)
    NUM_ID = 80
    NUM_EXP = 64
    BFM_MODEL = "model2017-1_face12_nomouth.h5"
    shape_pca, tex_pca, expr_pca, triangles = load_bfm_model(BFM_MODEL, NUM_ID, NUM_EXP)
    landmark_id = pd.read_csv('Landmarks68_model2017-1_face12_nomouth.anl', header=None).values.flatten().tolist()
    img = cv2.imread("1.jpg")
    y_gt = torch.tensor(detect_landmark(img), dtype=torch.float32)

    # Initialize the model
    model = LinearRegressionModel(n_id=NUM_ID, n_exp=NUM_EXP)
    # Define the optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD([model.alpha, model.delta, model.rotation, model.t], lr=0.001,  weight_decay=0.1)
    # Training loop
    NUM_EPOCHS = 500

    for epoch in range(NUM_EPOCHS):
        # Forward pass
        y_pred, ori_shape = model.forward(shape_pca, expr_pca, img)
        # y_pred = model.forward(shape_pca)
        # Compute the loss
        # print(y_gt, y_pred)
        reg = reg_loss(model)
        # loss = mse_loss(y_gt, y_pred[:, landmark_id, :2][0]) + 0.01 * reg
        # loss = 1/68 * mse_loss(y_gt, y_pred[:, landmark_id, :2][0])
        # loss = 1/68 * mse_loss(y_gt, y_pred[:, landmark_id, :2][0])  + 0.01 * reg
        loss = mse_loss(y_gt, y_pred[:, landmark_id, :2][0])  + 0.01 * reg

        # loss = 1.6e-3 * landmark_loss(y_pred[:, landmark_id, :2][0], y_gt) + 3.0e-4 * reg
        # loss = 1.6e-3 * y_pred
        # loss = mse_loss(y_pred[:, landmark_id, :2][0], y_gt)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print the loss every 100 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
        if (epoch + 1) % 20 == 0:
            img1 = np.copy(img)
            proj = y_pred.detach().cpu().numpy()[0]
            for p in proj:
                if (0 < p[0] < img.shape[1]) and (0 < p[1] < img.shape[0]):
                    img1[int(p[1])-1:int(p[1])+1, int(p[0])-1:int(p[0])+1] = (255, 0, 0)
            cv2.imwrite("debug/debug%04d.png" % epoch, img1)
    proj = y_pred.detach().cpu().numpy()[0]
    ori = ori_shape.detach().cpu().numpy()[0]
    rgb = []
    for p in proj:
        rgb.append(img[int(p[1]), int(p[0]),::-1])
    rgb = np.asarray(rgb, dtype=np.float32)
    output_mesh = trimesh.base.Trimesh(
            vertices=ori,
            faces=triangles,
            vertex_colors=rgb)
    output_mesh.export("out.obj")
    output_mesh.show()
