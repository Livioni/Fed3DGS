import os
from PIL import Image

# 图片所在的目录
image_dir = 'datasets/rubble-pixsfm/train/rgbs'

# TXT文件的路径

for i in range(15):  # 从00000到00019
    txt_filename = f"{i:05d}.txt"  # 生成格式化的文件名，如00000.txt, 00001.txt等
    txt_file_path = os.path.join('datasets/rubble-pixsfm_image_lists', txt_filename)  # 替换成你的TXT文件夹的实际路径

    # 读取TXT文件中的图片文件名
    with open(txt_file_path, 'r') as file:
        image_files = file.read().splitlines()

    total_size = 0

    # 遍历文件名，计算每个图片的大小
    for index, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        with Image.open(image_path) as img:
            total_size += os.path.getsize(image_path)
        # print("File_index, File Size:", index, os.path.getsize(image_path) / (1024 * 1024))

    total_size_mb = total_size / (1024 * 1024)
    print(f"Scene {i}, Total size of all images is: {total_size_mb:.2f} MB")
