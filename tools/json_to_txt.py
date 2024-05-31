import json

def create_files_from_json(json_file, file_prefix):
    # 读取JSON文件
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # 获取camera_list_in_region键的值，它是一个包含多个列表的列表
    camera_lists = data['camera_list_in_region']
    
    # 遍历每个子列表
    for index, camera_list in enumerate(camera_lists):
        # 对每个子列表的元素进行排序
        camera_list.sort()
        
        # 创建文件名，从00000.txt开始递增
        # file_name = file_prefix + f"{index:05d}/" +f"{index:05d}.txt"
        file_name = file_prefix +f"/{index:05d}.txt"
        
        # 打开文件并写入内容
        with open(file_name, 'w') as file:
            for camera_id in camera_list:
                file.write(f"{camera_id}.jpg\n")
    
    print(f"Files created: {len(camera_lists)} files.")

# 调用函数
json_file_path = 'outputs/vast_12clients_offical_colmap/rubble_vast_initial/block.json'  # 将 'path_to_your_json_file.json' 替换为你的 JSON 文件的实际路径
file_prefix = 'client_image_lists/vast_12clients_official_colmap'
create_files_from_json(json_file_path,file_prefix)