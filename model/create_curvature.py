'''
Based on the article "Robust Curvature Estimation and Geometry Analysis of3D point Cloud Surfaces"
'''
import numpy as np
from numpy.linalg import lstsq
from numpy.linalg import norm
import h5py
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.normals_Hough.python.lib.python import NormalEstimatorHough as NormalEstimator
import math, random

def fibonacci_sphere(samples=2048,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)


def get_knn(pc, num_nbrs):
    # +1 including the point itself
    nbrs = NearestNeighbors(n_neighbors=num_nbrs + 1, algorithm='auto', metric='euclidean').fit(pc)
    _, indices = nbrs.kneighbors(pc)  # (N, num_nbrs + 1)
    return indices[:, 1:]


# def get_curvature(pc, normals, num_nbrs):
#     num_points = pc.shape[0]
#     gauss_c = np.empty(num_points)
#     nbrs = get_knn(pc, num_nbrs)  # (num points, num_nbrs)
#     for i in range(num_points):
#         nbrs_normals = normals[nbrs[i, :], :]  # (num_nbrs, 3)
#         nbrs_coor = pc[nbrs[i, :], :]  # (num_nbrs, 3)
#         p_n = normals[i, :].reshape(3, 1)
#         phi = np.arccos(p_n[2])
#         psi = np.arctan(p_n[1] / p_n[0])
#         X = np.array([[-np.sin(psi).item(0), np.cos(psi).item(0), 0]])
#         Y = np.array([np.cos(psi)*np.cos(phi), np.cos(phi)*np.sin(psi), -np.sin(psi)])
#         trans_mat = np.row_stack([X.reshape(3), Y.reshape(3), p_n.reshape(3)])
#         # nbrs_normals = (trans_mat @ nbrs_normals.transpose()).transpose()
#         # nbrs_coor = (trans_mat @ nbrs_coor.transpose()).transpose()
#         nz = nbrs_normals[:, 2]
#         xi = nbrs_coor[:, 0]
#         yi = nbrs_coor[:, 1]
#         nx = nbrs_normals[:, 0]
#         ny = nbrs_normals[:, 1]
#         nxy = (xi * nx + yi * ny)/np.sqrt(xi ** 2 + yi ** 2)
#         kni = -nxy/(np.sqrt(nxy ** 2 + nz ** 2)*np.sqrt(xi ** 2 + yi ** 2))
#         print(kni)
#         proj_m = np.array([[1 - p_n[0] ** 2, -p_n[0]*p_n[1], -p_n[0]*p_n[2]],
#                            [-p_n[0]*p_n[1], 1 - p_n[1] ** 2, -p_n[1]*p_n[2]],
#                            [-p_n[0]*p_n[2], -p_n[1]*p_n[2], 1 - p_n[2] ** 2]]).reshape(3, 3)
#         nbrs_proj = proj_m @ nbrs_coor.transpose()  # (3, num_nbrs)
#         nbrs_theta = (X @ nbrs_proj).reshape(-1,)
#         m_matrix = np.column_stack([np.cos(nbrs_theta) ** 2,
#                                     2 * np.cos(nbrs_theta) * np.sin(nbrs_theta),
#                                     np.cos(nbrs_theta) ** 2])
#         mu, _, _, _ = lstsq(m_matrix, kni, rcond=None)
#         w = np.array([[mu[0], mu[1]],
#                       [mu[1], mu[2]]])
#         lambdas, _ = eigh(w)
#         gauss_c[i] = max([min([lambdas[0] * lambdas[1], 20]), -20])
#     return gauss_c

def get_curvature(pc, normals, num_nbrs):
    num_points = pc.shape[0]
    gauss_c = np.empty(num_points)
    nbrs = get_knn(pc, num_nbrs)  # (num points, num_nbrs)
    for i in range(num_points):
        nbrs_normals = normals[nbrs[i, :], :]  # (num_nbrs, 3)
        nbrs_coor = pc[nbrs[i, :], :]  # (num_nbrs, 3)
        p_n = normals[i, :].reshape(3, 1)
        if abs(p_n[0]) < abs(p_n[1]):
            u1 = np.cross(p_n.reshape(3), np.array([1, 0, 0]))
        else:
            u1 = np.cross(p_n.reshape(3), np.array([0, 1, 0]))
        u2 = np.cross(p_n.reshape(3), u1)
        trans_mat = np.row_stack([u1.reshape(3), u2.reshape(3)])
        proj_m = trans_mat.transpose() @ trans_mat
        nz = nbrs_normals[:, 2]
        xi = nbrs_coor[:, 0]
        yi = nbrs_coor[:, 1]
        nx = nbrs_normals[:, 0]
        ny = nbrs_normals[:, 1]
        nxy = (xi * nx + yi * ny)/np.sqrt(xi ** 2 + yi ** 2)
        kni = -nxy/(np.sqrt(nxy ** 2 + nz ** 2)*np.sqrt(xi ** 2 + yi ** 2))
        nbrs_coor = nbrs_coor
        nbrs_proj = proj_m @ nbrs_coor.transpose() - pc[i, :]  # (3, num_nbrs)
        assert (norm(nbrs_proj, axis=1, keepdims=True) > 1e-5).sum() < 1, f'norm:\n' \
                                                                          f'{norm(nbrs_proj, axis=1, keepdims=True)}\n ' \
                                                                          f'nbrscoor:\n{nbrs_coor}\n'\
                                                                          f'nbrsproj:\n{nbrs_proj.shape}'
        nbrs_proj = nbrs_proj / norm(nbrs_proj, axis=1, keepdims=True)
        cos_theta = (u1 @ nbrs_proj).reshape(-1,)
        # print(cos_theta)
        sin_theta = np.sqrt(1-cos_theta ** 2)
        m_matrix = np.column_stack([cos_theta ** 2,
                                    2 * cos_theta * sin_theta,
                                    sin_theta ** 2])
        mu, _, _, _ = lstsq(m_matrix, kni, rcond=None)
        w = np.array([[mu[0], mu[1]],
                      [mu[1], mu[2]]])
        lambdas, _ = eigh(w)
        gauss_c[i] = max([min([lambdas[0] * lambdas[1], 20]), -20])
    return gauss_c

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def plot_pc(pc, c=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = pc[:, 0], pc[:, 1], pc[:, 2]
    if c is not None:
        p = ax.scatter(xs, ys, zs, c=c)
    else:
        ax.scatter(xs, ys, zs)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    fig.colorbar(p)
    plt.show()


def plot_normals(pc, normals):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    u, v, w = normals[:, 0], normals[:, 1], normals[:, 2]

    ax.quiver(x, y, z, u, v, w, length=0.1)

    plt.show()


def calc_normals(pc, num_nbrs):
    estimator = NormalEstimator.NormalEstimatorHough()
    estimator.set_K(num_nbrs)
    pc = np.float64(pc)
    estimator.set_points(pc)
    estimator.estimate_normals()
    return np.float32(estimator.get_normals())


pc_file = '../data/modelnet40_ply_hdf5_2048/ply_data_train0.h5'
data, label = load_h5(pc_file)
normal_file = '../data/normals_5nbr_2048/normals_data_train0.h5'
index = 3
num_nbrs = 10
pc = data[index, :, :]
# pc = fibonacci_sphere()
pc_normals = calc_normals(pc, num_nbrs)
# pc_normals = pc
gauss_c = get_curvature(pc, pc_normals, num_nbrs)
# print(np.min(gauss_c))
plot_pc(pc, gauss_c, f'{label.item(0)}')

# plot_normals(pc, pc_normals)



