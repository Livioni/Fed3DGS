# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
import os
import random

import numpy as np
import torch

from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

def perform_kmeans(data, n_clusters):
    """
    使用 K-means 算法对三维坐标数据进行聚类。

    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, 3)，包含每个点的 x, y, z 坐标。
        n_clusters (int): 要分成的聚类数目。

    Returns:
        labels (np.ndarray): 每个数据点的聚类标签。
        centers (np.ndarray): 聚类的中心点坐标。
    """
    # 初始化 KMeans 模型
    kmeans = KMeans(n_clusters=n_clusters)
    
    # 对数据进行拟合和预测
    labels = kmeans.fit_predict(data)
    
    # 获取聚类中心点
    centers = kmeans.cluster_centers_
    
    return labels, centers

def count_points_in_clusters(labels, n_clusters):
    """
    计算每个聚类中的点的数量。

    Args:
        labels (np.ndarray): 每个数据点的聚类标签，由 K-means 聚类算法生成。
        n_clusters (int): 聚类的数量，应与 K-means 聚类时使用的数量相同。

    Returns:
        counts (np.ndarray): 每个聚类中的点数。
    """
    # 初始化一个数组，用于计数每个聚类的点数
    counts = np.zeros(n_clusters, dtype=int)
    
    # 计算每个聚类的点数
    for label in labels:
        counts[label] += 1
    
    return counts

def gen_client_data(c2ws: np.ndarray, n_data: int, camera_centers: np.ndarray = None):
    """
    Args:
        c2ws (np.ndarray): camera extrinsic (camera2world) that is
                        an ndarray of shape (#cameras, 3, 4).
                        the coordinate system is following mega-nerf,
                        i.e., (down, right, backward)
        n_data (int): a number of data for a client
                        
    Returns:
        indices (np.ndarray): client's data indices
    """
    n_cameras = c2ws.shape[0]
    xyz_coord = c2ws[:, :3, -1] # (#cameras, 3)
    
    if camera_centers is None:
        base_camera_idx = np.random.randint(n_cameras)
        center_xyz = xyz_coord[base_camera_idx]
    else:
        center_xyz = camera_centers
    dists = np.sum(np.square(xyz_coord - center_xyz), -1)
    
    indices = np.argsort(dists, 0)[:n_data]
    return np.sort(indices)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d',
                        required=True,
                        type=str,
                        help='/path/to/dataset')
    parser.add_argument('--output-dir', '-o',
                        required=True,
                        type=str,
                        help='/path/to/output_dir')
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='random seed')
    parser.add_argument('--n-clients',
                        default=200,
                        type=int,
                        help='number of clients')
    parser.add_argument('--n-pre-clients',
                        default=150,
                        type=int,
                        help='number of images for clients')
    parser.add_argument('--n-data-min', '-min',
                        default=125,
                        type=int,
                        help='minimum number of clients data')
    parser.add_argument('--n-data-max', '-max',
                        default=175,
                        type=int,
                        help='maximum number of clients data')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    fnames = sorted(os.listdir(os.path.join(args.dataset_dir, 'train/rgbs')))
    print('load metadatas')
    c2ws = []
    for fname in tqdm(fnames):
        c2ws.append(torch.load(os.path.join(args.dataset_dir, 'train/metadata', fname.split('.')[0] + '.pt'))['c2w'].numpy())
    c2ws = np.stack(c2ws)
    
    print('split data')
    os.makedirs(args.output_dir, exist_ok=True)
    
    xyz_coord = c2ws[:, :3, -1] # (#cameras, 3)
    labels, centers = perform_kmeans(xyz_coord, args.n_clients)
    cluster_counts = count_points_in_clusters(labels, args.n_clients)
    sample_camera_n = []
    for data in cluster_counts:
        if data <= args.n_data_min:
            sample_camera_n.append(data + np.random.randint(args.n_data_min - data, args.n_data_max - data))
        else:
            sample_camera_n.append(data)
            
    for i in range(args.n_clients):
        camera_centers = centers[i]
        indices = gen_client_data(c2ws, sample_camera_n[i], camera_centers)
        training_image_names = [fnames[idx] for idx in indices]
        np.savetxt(os.path.join(args.output_dir, str(i).zfill(5) + '.txt'), training_image_names, fmt="%s")
