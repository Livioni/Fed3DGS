import os
import glob

def calculate_ply_sizes(base_path, iterations):
    total_size_mb = 0
    for iteration in iterations:
        # 构造文件路径模式，用于匹配所有指定迭代的.ply文件
        pattern = os.path.join(base_path, '**', f'iteration_{iteration}', 'point_cloud.ply')
        # 使用glob模块查找所有匹配的文件
        files = glob.glob(pattern, recursive=True)
        for file in files:
            # 获取文件大小并累加
            total_size_mb += os.path.getsize(file) / (1024 * 1024)  # 转换为MB
    
    return total_size_mb

# 示例使用
base_path = 'outputs/rubble-pixsfm_local_models'
iterations = [200,400,600,800,1000,1200,1400,1600,1800,2000]
total_size = calculate_ply_sizes(base_path, iterations)
print(f'Total size of .ply files for iterations {iterations} is {total_size:.2f} MB')
