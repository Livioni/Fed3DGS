import os
import open3d as o3d
import numpy as np
import sys
import struct
from plyfile import PlyData, PlyElement

def read_file_to_list(filepath):
    """读取文本文件，并将内容转换为列表"""
    with open(filepath, 'r') as file:
        lines = file.readlines()
    # 移除每行末尾的换行符并返回
    return [line.strip() for line in lines]

def find_common_elements(filepath1, filepath2):
    """读取两个文本文件，找出并打印它们共有的元素数量"""
    # 读取文件内容到列表
    list1 = read_file_to_list(filepath1)
    list2 = read_file_to_list(filepath2)
    
    # 使用 np.intersect1d 找出两个列表中的共同元素
    common_elements = np.intersect1d(list1, list2)
    
    # 打印共同元素和它们的数量
    # print(f"Common elements: {common_elements}")
    print(f"Number of common elements: {len(common_elements)}")

    return len(common_elements)


def load_point_cloud(file_path):
    """尝试加载PLY文件作为点云数据"""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            raise ValueError("Loaded point cloud is empty.")
        return pcd
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        sys.exit(1)

def execute_icp(source_pcd, target_pcd, threshold):
    """执行ICP点云配准并返回结果"""
    trans_init = np.identity(4)  # 初始化为单位矩阵
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p

def execute_ransac_and_icp(source_pcd, target_pcd, threshold):
    """执行RANSAC和ICP点云配准并返回配准结果 格式与原ICP函数一致"""
    # 初始化变换矩阵
    trans_init = np.eye(4)
    
    # 设置RANSAC对齐参数
    ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=10000, confidence=0.95)
    
    # 执行RANSAC配准
    reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, 
        o3d.pipelines.registration.Feature(),  # 你需要在这里指定正确的特征匹配
        mutual_filter=True,
        max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        criteria=ransac_criteria
    )
    print("Coarse RANSAC alignment completed...")
    
    # 执行ICP配准
    reg_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, reg_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print("Fine ICP alignment completed...")
    
    return reg_icp



def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_points3D_text(path):
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1
    return xyzs, rgbs, errors

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def convert_ply(file_path):
    ply_path = os.path.join(file_path, "sparse/0/points3D.ply")
    bin_path = os.path.join(file_path, "sparse/0/points3D.bin")
    txt_path = os.path.join(file_path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    else:
        print("Using existing .ply file.")

def main():

    image_list_dir = 'client_image_lists/rubble-pixsfm_kmeans-10'
    colmap_dir = 'outputs/10clients/rubble-pixsfm_colmap_icp_results'
    interact_threshold = 10

    file_paths = [os.path.join(image_list_dir, f) for f in sorted(os.listdir(image_list_dir)) if f.endswith('.txt')]
    file_paths = file_paths[:32]
    
    # 比较每个文件与其后所有文件的交集
    for i in range(len(file_paths)):
        print(file_paths[i])
        list1 = read_file_to_list(file_paths[i])

        number_part = os.path.basename(file_paths[i]).split('.')[0]
        target_path = os.path.join(colmap_dir, number_part)

        for j in range(i + 1, len(file_paths)):
            list2 = read_file_to_list(file_paths[j])
            common_elements = np.intersect1d(list1, list2)
            print(len(common_elements))
            
            # 检查交集长度是否超过阈值
            if len(common_elements) > interact_threshold:
                number_part_ = os.path.basename(file_paths[j]).split('.')[0]
                source_path = os.path.join(colmap_dir, number_part_)

                print(source_path)
                print(target_path)

                convert_ply(source_path)
                convert_ply(target_path)

                source_pcd_path = os.path.join(source_path, "sparse/0/points3D.ply")
                target_pcd_path = os.path.join(target_path, "sparse/0/points3D.ply")

                source_pcd = load_point_cloud(source_pcd_path)
                target_pcd = load_point_cloud(target_pcd_path)

                threshold = 0.02  # 配准的最大容许距离
                icp_result = execute_icp(source_pcd, target_pcd, threshold)
                # icp_result = execute_ransac_and_icp(source_pcd, target_pcd, threshold)

                source_pcd.transform(icp_result.transformation)
                print("ICP registration finished...")

                # 构建输出文件路径
                output_file_path = os.path.join(source_path, "sparse/0/points3D.ply")
                o3d.io.write_point_cloud(output_file_path, source_pcd)
                print(f"The transformed point cloud is saved to {output_file_path}")

class RedirectLogger(object):
    def __init__(self, path):
        # 打开文件时使用追加模式
        self.log_file = open(path, "a")

    def write(self, message):
        self.log_file.write(message)

    def flush(self):
        self.log_file.flush()

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self.log_file.close()


if __name__ == "__main__":
    # log_file_path = "icp_registration_log.txt"
    # with RedirectLogger(log_file_path):
    #     if len(sys.argv) < 3:
    #         print("Usage: python icp_registration.py <source_path> <target_path>")
    #         print("---------------------------------------------")
    #         sys.exit("Exiting due to insufficient arguments.")
    main()
    print("ICP registration finished.")
    print("---------------------------------------------")
