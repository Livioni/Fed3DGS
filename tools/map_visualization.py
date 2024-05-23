import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    parser.add_argument('--image-list-dir', '-l',
                        required=True,
                        type=str,
                        help='/path/to/image-list-dir')
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
    xy_coord = c2ws[:, :2, -1] # (#cameras, 3)
    normalized_data = (xy_coord - xy_coord.min(axis=0)) / (xy_coord.max(axis=0) - xy_coord.min(axis=0))

    # 绘制规范化后的数据
    plt.scatter(normalized_data[:, 0], normalized_data[:, 1], alpha=0.6)
    plt.title('Normalized Coordinates Visualization')
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'normalized_coordinates.png'), dpi=300)
    plt.close()
    
    # draw 
    txt_files = os.listdir(args.image_list_dir)

    specified_txt_file = '00000.txt'
    name = specified_txt_file.split('.')[0]
    # 为了便于演示，我们创建一个图形和一个坐标轴
    fig, ax = plt.subplots()


    # 读取并处理特定的txt文件
    with open(os.path.join(args.image_list_dir, specified_txt_file), 'r') as file:
        indices = []
        for line in file:
            # 解析每行数据以获取图片编号，假设格式是'000172.jpg'
            index = int(line.strip().split('.')[0])
            indices.append(index)
        
        # 根据索引获取这些点的坐标
        points = xy_coord[indices]
        
        # 特定文件中的点变为红色，并显示索引
        for idx in range(len(xy_coord)):
            point = xy_coord[idx]
            if idx in indices:
                ax.scatter(point[0], point[1], color='red', alpha=0.6)
            else:
                ax.scatter(point[0], point[1], alpha=0.5, edgecolor='blue', facecolors = 'none')
            ax.text(point[0], point[1], str(idx), color="k", fontsize=3)
        
        # 计算这些点的最小外接矩形（简单示例）
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        
        # 画矩形
        rect = plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    # 设置标题和坐标轴标签
    ax.set_title('Points with Bounding Boxes from Specified txt File')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'client_{name}.png'), dpi=300)
    plt.close()
        