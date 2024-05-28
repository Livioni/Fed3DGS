
import os
import sys
import uuid
import json
from random import randint
from PIL import Image
import multiprocessing as mp
from multiprocessing import Process, current_process
from typing import List, Dict, Any
import logging
import torch
import numpy as np
from tqdm import tqdm
import argparse
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, PILtoTorch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from progressively_build_global_model import check_buffer, get_model_params, distillation
from utils.model_update_utils import (get_model_params,
                                      compute_visible_point_mask,
                                      get_cameras_from_metadata,)    
from eval import evaluation

def update_model(client_model_index : str,
                 global_params: Dict[str, Any],
                 client_model: GaussianModel,
                 client_metadatas: List[Dict[str, Any]],
                 global_model_camera_meta: List[Dict[str, Any]],
                 min_opacity: float,
                 lr_opacity: float,
                 lr_mlp: float,
                 wd_mlp: float,
                 lr_hash: float,
                 lr_avec: float,
                 n_epoch: int,
                 bg_color: torch.Tensor,
                 resolution_scale: int=1,
                 far: int=100):
    # get camera intrinsic
    image_height, image_width, fovx, fovy, viewmats = get_cameras_from_metadata(client_metadatas) #获取client model的相机参数
    # fovx: 水平视场角 fovy: 垂直视场角 viewmats: 相机视图矩阵
    image_height = list(map(lambda x: x //resolution_scale, image_height))
    image_width = list(map(lambda x: x //resolution_scale, image_width))
    # get visible Gaussians
    vis_msk = compute_visible_point_mask(global_params['xyz'], client_metadatas, 'cpu') #在client model的视角下，global model中的点是否可见
    print(f"#global model's points: {len(global_params['xyz'])} ({vis_msk.sum()} visible points)")
    print(f"#local model's points: {len(client_model._xyz.data)}")
    xyz_g = global_params['xyz']
    rot_g = global_params['rotation']
    scale_g = global_params['scaling']
    opacity_g = global_params['opacity']
    sh_g = torch.cat([global_params['features_dc'], global_params['features_rest']], 1)
    vis_xyz_g = xyz_g[vis_msk] #global model中可见的点
    vis_rot_g = rot_g[vis_msk] #global model中可见的点的旋转矩阵
    vis_scale_g = scale_g[vis_msk] #global model中可见的点的缩放矩阵
    vis_opacity_g = opacity_g[vis_msk] #global model中可见的点的透明度
    vis_sh_g = sh_g[vis_msk] #global model中可见的点的球谐函数
    tmp_global_model = GaussianModel(client_model.max_sh_degree)
    print(f'#points before model update: {len(vis_xyz_g)}')
    new_params = dict(xyz=vis_xyz_g,
                      rotation=vis_rot_g,
                      scaling=vis_scale_g,
                      features_dc=vis_sh_g[:, :1],
                      features_rest=vis_sh_g[:, 1:],
                      opacity=vis_opacity_g,
                      app_mlp=global_params['app_mlp'],
                      app_pos_emb=global_params['app_pos_emb'])
    tmp_global_model.set_params(new_params)

    tmp_global_model = distillation(tmp_global_model,
                                    client_model,
                                    global_model_camera_meta,
                                    image_height,
                                    image_width,
                                    fovx,
                                    fovy,
                                    viewmats,
                                    bg_color,
                                    lr_opacity,
                                    lr_mlp,
                                    wd_mlp,
                                    lr_hash,
                                    lr_avec,
                                    resolution_scale,
                                    n_epoch,
                                    min_opacity,
                                    far=far)

    vis_xyz_g, vis_rot_g, vis_scale_g, vis_opacity_g, vis_sh_g = get_model_params(tmp_global_model, preact=True, device='cpu')
    app_mlp = tmp_global_model.mlp.state_dict()
    app_pos_emb = tmp_global_model.pos_emb.state_dict()
    
    # TODO partition local model
    local_model_points = len(client_model._xyz.data)
    xyz_l = vis_xyz_g[-local_model_points:]
    rot_l = vis_rot_g[-local_model_points:]
    scale_l = vis_scale_g[-local_model_points:]
    opacity_l = vis_opacity_g[-local_model_points:]
    sh_l = vis_sh_g[-local_model_points:]
    
    # client 0 params
    if client_model_index == '00001':
        client0_points = len(xyz_g)
        xyz_0 = torch.cat([xyz_g[~vis_msk], vis_xyz_g])[:client0_points]
        rot_0 = torch.cat([rot_g[~vis_msk], vis_rot_g])[:client0_points]
        scale_0 = torch.cat([scale_g[~vis_msk], vis_scale_g])[:client0_points]
        opacity_0 = torch.cat([opacity_g[~vis_msk], vis_opacity_g])[:client0_points]
        sh_0 = torch.cat([sh_g[~vis_msk], vis_sh_g])[:client0_points]

        client0_params = dict(xyz=xyz_0,
                              rotation=rot_0 ,
                              scaling=scale_0,
                              features_dc=sh_0[:, :1],
                              features_rest=sh_0[:, 1:],
                              opacity=opacity_0,
                              app_mlp=app_mlp,
                              app_pos_emb=app_pos_emb)
    else:
        client0_params = None
        
    # prune points
    prune_mask = (vis_opacity_g.sigmoid() > min_opacity).reshape(-1)
    if prune_mask.any():
        print(f'prune {(~prune_mask).sum()} points')
        vis_xyz_g = vis_xyz_g[prune_mask]
        vis_rot_g = vis_rot_g[prune_mask]
        vis_scale_g = vis_scale_g[prune_mask]
        vis_opacity_g = vis_opacity_g[prune_mask]
        vis_sh_g = vis_sh_g[prune_mask]
    xyz_g = torch.cat([xyz_g[~vis_msk], vis_xyz_g])
    rot_g = torch.cat([rot_g[~vis_msk], vis_rot_g])
    scale_g = torch.cat([scale_g[~vis_msk], vis_scale_g])
    opacity_g = torch.cat([opacity_g[~vis_msk], vis_opacity_g])
    sh_g = torch.cat([sh_g[~vis_msk], vis_sh_g])

    new_params = dict(xyz=xyz_g,
                      rotation=rot_g,
                      scaling=scale_g,
                      features_dc=sh_g[:, :1],
                      features_rest=sh_g[:, 1:],
                      opacity=opacity_g,
                      app_mlp=app_mlp,
                      app_pos_emb=app_pos_emb)
    
    local_params = dict(xyz=xyz_l,
                        rotation=rot_l,
                        scaling=scale_l,
                        features_dc=sh_l[:, :1],
                        features_rest=sh_l[:, 1:],
                        opacity=opacity_l,
                        app_mlp=app_mlp,
                        app_pos_emb=app_pos_emb)
                        
    print(f'#points after model update: {len(xyz_g)}')
    return new_params, local_params, client0_params

def _update_model(global_params, client_model_index, metadatas, client_metadatas, global_model_cam_list, intersection, bg_color, load_iter, args):
    # load local model
    client_model_file = os.path.join(args.model_dir,
                                     client_model_index,
                                     'point_cloud/iteration_' + str(load_iter) + '/point_cloud.ply')
    client_model = GaussianModel(args.sh_degree) #初始化一个client model
    client_model.load_ply(client_model_file) #加载client model
    print(f'update model with {client_model_index}-th clients at {load_iter}-th iteration')
    g_sub_l = np.setdiff1d(global_model_cam_list, intersection) #取global model和client model的差集,在global model中但不在client model中的图片
    global_model_camera_meta = [metadatas[fname.split('.')[0]] for fname in g_sub_l] #取出g_sub_l中的图片对应的metadata
    global_params, local_params, client0_params = update_model(client_model_index, global_params, client_model, client_metadatas,
                                               global_model_camera_meta, args.min_opacity, args.lr_opacity,
                                               args.lr_mlp, args.wd_mlp, args.lr_hash, args.lr_avec,
                                               args.n_kd_epoch, bg_color, args.resolution, far=args.far)
    return global_params, local_params, client0_params

def aggregate_global_model(args, load_iter):
    metadata_dir = os.path.join(args.dataset_dir, 'train/metadata')
    print('load metadata')
    # load metadata including camera intrinsic and extrinsic
    metadata_files = sorted(os.listdir(metadata_dir))
    metadatas = {}
    for fname in tqdm(metadata_files):
        file_idx = fname.split('.')[0]
        metadatas[file_idx] = torch.load(os.path.join(metadata_dir, fname)) #一个字典，对应图片index和对应的metadata
    # load image indices in clients data
    print('load image lists')
    index_files = sorted(os.listdir(args.index_dir)) #列出了index文件夹下的所有图片索引txt
    if args.shuffle:
        index_files = list(np.random.permutation(index_files))
    if args.n_clients > 0:
        index_files = index_files[:args.n_clients]
    image_lists = [list(np.loadtxt(os.path.join(args.index_dir, fname), dtype=str))
                   for fname in index_files if '.txt' in fname] 
    # load a 0-th local model as a global model
    print('initialize global model')
    # seed_model_index = index_files.pop(0).split('.')[0] #取出第一个client的index 即00000
    seed_model_index = index_files[0].split('.')[0]
    seed_model_file = os.path.join(args.model_dir,
                                   seed_model_index,
                                   'point_cloud/iteration_' + str(load_iter) + '/point_cloud.ply')
    global_model = GaussianModel(args.sh_degree) #初始化一个global model
    global_model.load_ply(seed_model_file) #加载第一个client的模型
    # get model params
    xyz_g, rot_g, scale_g, opacity_g, sh_g = get_model_params(global_model, preact=True, device='cpu') #获取global model的参数
    # xyz_g: 3D坐标 rot_g: 旋转矩阵 scale_g: 缩放矩阵 opacity_g: 透明度 sh_g: 球谐函数
    global_params = dict(xyz=xyz_g,
                        rotation=rot_g,
                        scaling=scale_g,
                        features_dc=sh_g[:, :1],
                        features_rest=sh_g[:, 1:],
                        opacity=opacity_g,
                        app_mlp=global_model.mlp.state_dict(),
                        app_pos_emb=global_model.pos_emb.state_dict())

    del global_model
    # global model's camera list
    global_model_cam_list = image_lists[0]
    # placeholder
    client_buffer = []
    # set background color
    bg_color = torch.Tensor([1., 1., 1.]).cuda() if args.white_background else torch.Tensor([0., 0., 0.]).cuda()
    n_added_client = 1
    
    for client_idx, client_cam_list in zip(index_files, image_lists):
        if client_idx == index_files[0]:
            continue
        intersection = np.intersect1d(global_model_cam_list, client_cam_list) #取global model和client model的交集
        # client selection
        if len(intersection) < args.overlap_img_threshold:
            client_buffer.append([client_idx, client_cam_list])
            continue
        print('---------------')
        # load a local model
        client_model_index = client_idx.split('.')[0]
        client_metadatas = [metadatas[fname.split('.')[0]] for fname in client_cam_list]
        print('Merge client model: ', client_model_index)
        global_params, local_params, client0_params = _update_model(global_params, client_model_index, metadatas, client_metadatas,
                                    global_model_cam_list, intersection, bg_color, load_iter, args)
        # update global model's camera list
        global_model_cam_list = np.union1d(global_model_cam_list, client_cam_list)
        n_added_client += 1
        
        # save model
        if load_iter != args.iterations:
            if (n_added_client % args.save_freq) == 0:
                torch.save(global_params, os.path.join(args.output_dir, f'chkpnt_{n_added_client}_updated.pth'))
                
        if load_iter != args.iterations:
            # save local model
            torch.save(local_params, os.path.join(args.model_dir, client_model_index, f'chkpnt_{load_iter}_updated.pth'))
            if client_idx == '00001.txt':
                torch.save(client0_params, os.path.join(args.model_dir, '00000', f'chkpnt_{load_iter}_updated.pth'))
            
        # aggregate buffered models
        while True:
            global_params, client_buffer, global_model_cam_list, updated, n_added_client = check_buffer(global_params, client_buffer, metadatas,
                                                                                                        load_iter, bg_color, global_model_cam_list,
                                                                                                        n_added_client, args)
            if not updated:
                break

    print("Save global model at {} iteration".format(load_iter))
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created directory: {args.output_dir}")
        except Exception as e:
            print(f"Error creating directory {args.output_dir}: {e}")
        
    torch.save(global_params, os.path.join(args.output_dir, f'global_model_epoch{load_iter}.pth'))
        
    return

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, local_model_checkpoint_path, logger, debug_from = -1):
    first_iter = 0
    print("Start training client {} at iteration {}".format(local_model_checkpoint_path, first_iter))
    tb_writer = prepare_output_and_logger(dataset,logger)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, logger=logger)
    gaussians.training_setup(opt)
    ## 初始化高斯点
    if checkpoint: # Load checkpoint
        updated_params = torch.load(checkpoint, map_location="cuda")
        (local_params, first_iter) = torch.load(local_model_checkpoint_path)
        gaussians.restore(local_params, opt)
        gaussians.set_rotation(updated_params['rotation'])
        gaussians.set_opacity(updated_params['opacity'])
        gaussians.set_mlp(updated_params['app_mlp'])
        gaussians.set_pos_emb(updated_params['app_pos_emb'])
        gaussians.reset_setting()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True) #用于精准记录GPU上的训练时间
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy() #将该client的训练数据拷贝一份，包括各个视角的图片和相机参数
    training_data_idx = None
    ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record() #记录开始时间

        gaussians.update_learning_rate(iteration) #根据迭代次数更新学习率

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not training_data_idx:
            training_data_idx = torch.arange(len(viewpoint_stack)).tolist() #生成训练图片的索引
        sampled_idx = randint(0, len(training_data_idx)-1) #随机选择一个图片
        train_idx = training_data_idx.pop(sampled_idx) #将该图片从训练集中移除
        viewpoint_cam = viewpoint_stack[train_idx] #选择该图片的相机参数

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, train_idx, background)
        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["render"], #渲染结果
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])

        # Loss
        gt_image = viewpoint_cam.original_image #真实图片 torch.Size([3, 864, 1152])
        if isinstance(gt_image, torch.Tensor):
            gt_image = gt_image.cuda()
        else:
            gt_image = PILtoTorch(Image.open(gt_image), (viewpoint_cam.image_width, viewpoint_cam.image_height)).cuda()[:3, ...]
        
        if viewpoint_cam.is_val: # remove right-side pixels
            gt_image = gt_image[..., :gt_image.shape[-1]//2]
            image = image[..., :image.shape[-1]//2]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        iter_end.record() #记录一次结束时间

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log #指数移动平均
            # if iteration % 10 == 0:
            #     progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"}) #设置进度条的显示内容
            #     progress_bar.update(10) #更新进度条
            # if iteration == opt.iterations:
            #     progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, -1, background), logger=logger)
            
            if (iteration in saving_iterations):
                if logger:
                    logger.info(f"Saving Gaussians at iteration {iteration}")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
            if (iteration in checkpoint_iterations):
                if logger:
                    logger.info(f"Saving Checkpoint at iteration {iteration}")
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) #更新最大半径
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) #更新高斯点的位置和半径

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: #每隔densification_interval次迭代进行一次稠密化
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None #稠密化的阈值
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) #稠密化和剪枝

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter): #每隔opacity_reset_interval次迭代重置透明度
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()  #更新高斯点的位置和半径
                gaussians.optimizer.zero_grad(set_to_none = True) #清空梯度
                if gaussians.appearance_optim is not None:
                    gaussians.appearance_optim.step()
                    gaussians.appearance_optim.zero_grad(set_to_none=True)
    return


def prepare_output_and_logger(args,logger):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    logger.info("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, logger = None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    if isinstance(viewpoint.original_image, torch.Tensor):
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(PILtoTorch(Image.open(viewpoint.original_image), (viewpoint.image_width, viewpoint.image_height)).cuda()[:3, ...], 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])       
                if logger:
                    logger.info("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))   
                else:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def setup_logging(process_id, file_path):
    # 创建一个 logger
    logger = logging.getLogger(f'Client_{process_id}')
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建文件 handler，用于写入日志文件
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_handler = logging.FileHandler(f'{file_path}/client_{process_id}.log')

    # 创建 formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 添加 handler 到 logger
    logger.addHandler(file_handler)

    return logger

def parallel_local_training(gpu_id, client_index, lp_args, op_args, pp_args, test_iterations, save_iterations, checkpoint_iterations, start_checkpoint):
    torch.cuda.set_device(gpu_id)
    
    # modify the source path and model path
    source_path = lp_args.source_path
    model_path = lp_args.model_path
    client_model_path = f"{model_path}/{client_index:05d}"
    client_source_path = f"{source_path}/{client_index:05d}"
    lp_args.source_path = client_source_path
    lp_args.model_path = client_model_path
    op_args.iterations = save_iterations
    
    save_iterations = [save_iterations]
    checkpoint_iterations = [checkpoint_iterations]
    if start_checkpoint:
        local_model_checkpoint_path = f"{model_path}/{client_index:05d}/chkpnt{start_checkpoint}.pth"
        start_checkpoint_path = f"{model_path}/{client_index:05d}/chkpnt_{start_checkpoint}_updated.pth"
    else:
        start_checkpoint_path = None
        local_model_checkpoint_path = None
        
    
    logger = setup_logging(client_index,file_path=client_model_path)
    # 启动训练
    logger.info("Starting process")
    training(lp_args, op_args, pp_args, test_iterations, save_iterations, checkpoint_iterations, start_checkpoint_path, local_model_checkpoint_path, logger)
    logger.info("Finishing process")

def delete_specific_pth_files(directory, filename):
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file matches the specific filename
            if file == filename:
                # Construct the full file path
                file_path = os.path.join(root, file)
                try:
                    # Remove the file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def setup_args():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")

    # ModelParams, OptimizationParams, PipelineParams integration
    # Assuming these methods correctly prefix their arguments to avoid conflicts
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Local training params
    parser.add_argument('--debug-from', type=int, default=-1)
    parser.add_argument('--detect-anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument('--index-dir', required=True, type=str,
                        help='/path/to/image-list-file-dir')
    parser.add_argument('--model-dir', required=True, type=str,
                        help='/path/to/client-model-dir')
    parser.add_argument('--eval-out', required=True, type=str,
                        help='/path/to/eval-dir')

    # Global training params
    parser.add_argument('--output-dir', '-o', default='./', type=str,
                        help='/path to output dir')
    parser.add_argument('--dataset-dir', '-data', required=True, type=str,
                        help='/path to dataset dir')
    # Experimental setup
    parser.add_argument('--shuffle', action='store_true',
                        help='If True, randomly aggregating client models')
    # Model args
    parser.add_argument('--sh-degree', default=3, type=int)

    # Alignment args
    parser.add_argument('-lr', default=1e-3, type=float,
                        help='Learning rate for alignment')
    parser.add_argument('--overlap-img-threshold', '-oth', default=0, type=int)

    # Aggregation args
    parser.add_argument('--min-opacity', '-min-o', default=0.005, type=float)
    parser.add_argument('--n-clients', default=-1, type=int)
    parser.add_argument('--n-kd-epoch', default=5, type=int)

    # Optimizer args
    parser.add_argument('--lr-opacity', '-lro', default=0.05, type=float)
    parser.add_argument('--lr-mlp', '-lrm', default=1e-4, type=float)
    parser.add_argument('--wd-mlp', default=1e-4, type=float)
    parser.add_argument('--lr-hash', '-lrh', default=1e-4, type=float)
    parser.add_argument('--lr-avec', default=1e-3, type=float)

    # Misc
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--save-freq', default=100, type=int)
    parser.add_argument('--far', default=100, type=int)
    parser.add_argument('--n-iter', default=100, type=int)

    return parser.parse_args(), lp, op, pp

if __name__ == "__main__":
    # Set up command line argument parser
    args, lp, op, pp = setup_args()

    # args.save_iterations.append(args.iterations)
    print('Initialize local model training')

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    #Parallel training
    mp.set_start_method('spawn', force=True)
    
    # prepare logging
    cuda_devices = torch.cuda.device_count()
    print(f"Found {cuda_devices} CUDA devices")
    trainin_round = args.clients // cuda_devices
    test_iterations = list(range(100, 20001, 200))

    # Epochs Settings
    aggregate_iterations = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    save_iteration_pools = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    checkpoint_iteration_pools = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    start_checkpoint_pools = [None, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]
    
    # Main Loops
    for epoch in range(len(aggregate_iterations)):
        print(f"Start training epoch {epoch}")
        save_iterations = save_iteration_pools[epoch] # 用于local model 保存ply文件
        checkpoint_iterations = checkpoint_iteration_pools[epoch] # 用于local model 保存checkpoint .pth
        start_checkpoint = start_checkpoint_pools[epoch] # 用于 local model 加载checkpoint .pth 开训 
        print(f"Start training local models at {start_checkpoint}, save at {checkpoint_iterations}, end at {save_iterations}")

        for i in range(trainin_round):
            client_index_1 = i
            client_index_2 = trainin_round + i
            
            # Debug
            # parallel_local_training(0, client_index_1, lp.extract(args), op.extract(args), pp.extract(args),
            #                         test_iterations, save_iterations, checkpoint_iterations, start_checkpoint)
            
            processes = []
            p1 = Process(target=parallel_local_training, name = f"Client_{client_index_1}",
                        args=(0, client_index_1, lp.extract(args), op.extract(args), pp.extract(args), 
                        test_iterations, save_iterations, checkpoint_iterations, start_checkpoint))
            
            p2 = Process(target=parallel_local_training, name = f"Client_{client_index_2}",
                        args=(1, client_index_2, lp.extract(args), op.extract(args), pp.extract(args), 
                        test_iterations, save_iterations, checkpoint_iterations, start_checkpoint))
                                
            p1.start()
            p2.start()

            processes.append(p1)
            processes.append(p2)
            
            for p in processes:
                p.join()
                    
            torch.cuda.empty_cache()
            print("local client {} and {} training finished".format(client_index_1, client_index_2))
            print("###############################################")
        
        # Specify the filename you want to delete
        local_model_pth = f"chkpnt{start_checkpoint}.pth"
        updated_model_pth = f"chkpnt_{start_checkpoint}_updated.pth"
        delete_specific_pth_files(args.model_dir, local_model_pth)
        delete_specific_pth_files(args.model_dir, updated_model_pth)

        #################################################
        print("Start to aggregate the local models")
        aggregate_global_model(args, save_iterations)
        torch.cuda.empty_cache()


        print("Start to evaluate the global model")
        output_dir = os.path.join(args.eval_out, f"fed_eval_epoch{save_iterations}")
        os.makedirs(output_dir, exist_ok=True)

        # setup logger
        logger = logging.getLogger('eval')
        if logger.hasHandlers():
            logger.handlers.clear()  # 清除之前的handlers
        logger.setLevel(logging.INFO)
        s_handler = logging.StreamHandler(stream=sys.stdout)
        plain_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(logging.INFO)
        logger.addHandler(s_handler)
        f_handler = logging.FileHandler(os.path.join(output_dir, 'console.log'))
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.INFO)
        logger.addHandler(f_handler)

        global_model_path = os.path.join(args.output_dir, f"global_model_epoch{save_iterations}.pth")
        logger.info(f'load global model from {global_model_path}')
        global_params = torch.load(global_model_path)
        logger.info(f'#Gaussians {len(global_params["xyz"])}')
        logger.info('load metadata')
       
        # set background color
        bg_color = torch.Tensor([1., 1., 1.]).cuda() if args.white_background else torch.Tensor([0., 0.,0.]).cuda()
        
        # evaluation
        val_image_lists = sorted(os.listdir(os.path.join(args.dataset_dir, 'val/rgbs')))
        val_metadatas = [torch.load(os.path.join(args.dataset_dir, 'val/metadata', f.split('.')[0]+'.pt')) for f in val_image_lists]
        images, depths, psnr, ssim, lpips = evaluation(global_params,
                                                        args.dataset_dir,
                                                        val_image_lists,
                                                        val_metadatas,
                                                        bg_color,
                                                        args.sh_degree,
                                                        args.n_iter,
                                                        args.lr,
                                                        args.resolution)

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(dict(psnr=psnr, ssim=ssim, lpips=lpips), f)

        for fname, img in zip(val_image_lists, images):
            save_image(img, os.path.join(output_dir, fname))

        for fname, img in zip(val_image_lists, depths):
            plt.imsave(os.path.join(output_dir, 'depth-' + fname), img)

        
                
            
            
        
        
        



