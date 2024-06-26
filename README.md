## 1Logs

- 2024.05.17-0 init
- 2024.05.17-1 使用Kmeans聚15类划分，并保证所有图片全覆盖，大约相邻范围内有10-40左右的重叠视角
- 2024.05.17-2 单机双卡训练
- 2024.05.17-3/4 开始测试15类全覆盖的client的精度
- 2024.05.20-1 注解代码1
- 2024.05.21-1 尝试15个clients，没有发现明显规律
- 2024.05.21-2 重新划分了20个clients，并且有大量重叠图片; 同步至中关村
- 2024.05.22-1 修复性能损失。
- 2024.05.23-1 完成Real Fed 3DGS
- 2024.05.24-0 同步
- 2024.05.25-0 更新联邦学习代码，发现效果已达上限。
- 2024.05.27-0 怀疑是client数量问题，并且添加ICP
- 2024.05.27-1 同步至xns开始调试
- 2024.05.28-0 修改gaussian-splatting 至 gaussian_splatting
- 2024.05.28-1 删除submodule 

export CUDA_VISIBLE_DEVICES=1

export CUDA_VISIBLE_DEVICES=0

## Training Local Models:

```bash
python tools/gen_client_data.py -d datasets/rubble-pixsfm \
                                -o datasets/rubble-pixsfm_image_lists \
                                --n-clients 20 --n-data-max 180 
```

## Distrubuted Training

### Colmap Pre-process

```bash
bash scripts/client_training.sh 5 9 outputs/10clients/rubble-pixsfm_colmap_results \
                                     datasets/rubble-pixsfm \
                                     client_image_lists/rubble-pixsfm_kmeans-10 \
                                     outputs/rubble-pixsfm_local_models
```

```bash
bash scripts/device_0_training.sh 0 9 outputs/20clients/rubble-pixsfm_colmap_results \
                                      datasets/rubble-pixsfm \
                                      client_image_lists/rubble-pixsfm_kmeans-20 \
                                      outputs/20clients/rubble-pixsfm_local_models
```

```bash
bash scripts/device_1_training.sh 10 19 outputs/20clients/rubble-pixsfm_colmap_results \
                                        datasets/rubble-pixsfm \
                                        client_image_lists/rubble-pixsfm_kmeans-20 \
                                        outputs/20clients/rubble-pixsfm_local_models
```

## Single Scene Training

```bash
python gaussian_splatting/train.py -s outputs/15clients/rubble-pixsfm_colmap_results/00007 \
                                   -i datasets/rubble-pixsfm/train/rgbs \
                                   -w \
                                   -m outputs/test/00007
```

## Build Global Model

```bash
python gaussian_splatting/build_global_model.py \
                                                -w -o outputs/20clients/global_models \
                                                -m outputs/20clients/rubble-pixsfm_local_models  \
                                                -i client_image_lists/rubble-pixsfm_kmeans-20 \
                                                -data datasets/rubble-pixsfm \
                                                --sh-degree 3
```

## Progressively_Build

```bash
python gaussian_splatting/progressively_build_global_model.py \
                                                -w -o outputs/20clients/global_models/single4_000 \
                                                -m outputs/20clients/rubble-pixsfm_local_models  \
                                                -i client_image_lists/rubble-pixsfm_kmeans-20 \
                                                -data datasets/rubble-pixsfm \
                                                --sh-degree 3
```

## Evaluation

```bash
python gaussian_splatting/eval.py -w -o eval/kmeans-10-20_000 -g outputs/global_model/kmeans-10-20_000/global_model_epoch20000.pth -data datasets/rubble-pixsfm
```

## RealFed

```bash
python gaussian_splatting/realfed.py  \
-s outputs/10clients/rubble_colmap_results_icp \
-i datasets/rubble-pixsfm/train/rgbs \
-w -m outputs/10clients/real_fed_models \
-o outputs/10clients/real_fed_global_models \
-data datasets/rubble-pixsfm \
--index-dir client_image_lists/rubble-pixsfm_kmeans-10 \
--model-dir outputs/10clients/real_fed_models \
--eval-out  eval \
--clients 10 \
--overlap-img-threshold 0
```
