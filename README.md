
## Logs

- 2024.05.17-0 init 
- 2024.05.17-1 使用Kmeans聚15类划分，并保证所有图片全覆盖，大约相邻范围内有10-40左右的重叠视角
- 2024.05.17-2 单机双卡训练
- 2024.05.17-3/4 开始测试15类全覆盖的client的精度
- 2024.05.20-1 注解代码1
- 2024.05.21-1 尝试15个client，没有发现明显规律


export CUDA_VISIBLE_DEVICES=1 
export CUDA_VISIBLE_DEVICES=0

## Training Local Models:
```bash
python tools/gen_client_data.py -d datasets/rubble-pixsfm \
                                -o datasets/rubble-pixsfm_image_lists \
                                --n-clients 15 --n-data-max 150 
```

## Distrubuted Training

### Colmap Pre-process
``` bash
bash scripts/client_training.sh 0 14 outputs/rubble-pixsfm_colmap_results \
                                     datasets/rubble-pixsfm \
                                     client_image_lists/rubble-pixsfm_kmeans-15 \
                                     outputs/rubble-pixsfm_local_models
```


``` bash
bash scripts/device_0_training.sh 0 7 outputs/rubble-pixsfm_colmap_results \
                                     datasets/rubble-pixsfm \
                                     client_image_lists/rubble-pixsfm_kmeans-15 \
                                     outputs/rubble-pixsfm_local_models
```

``` bash
bash scripts/device_1_training.sh 8 14 outputs/rubble-pixsfm_colmap_results \
                                     datasets/rubble-pixsfm \
                                     client_image_lists/rubble-pixsfm_kmeans-15 \
                                     outputs/rubble-pixsfm_local_models
```

## Single Scene Training

``` bash
python gaussian-splatting/train.py -s outputs/rubble-pixsfm_colmap_results/00004 \
                                   -i datasets/rubble-pixsfm/train/rgbs \
                                   -w \
                                   -m outputs/rubble-pixsfm_local_models/00004
```

## Build Global Model

``` bash
python gaussian-splatting/build_global_model.py \
                                                -w -o outputs/global_model \
                                                -m outputs/rubble-pixsfm_local_models  \
                                                -i client_image_lists/rubble-pixsfm_k_means \
                                                -data datasets/rubble-pixsfm \
                                                --sh-degree 3
```

## Progressively_Build

``` bash
python gaussian-splatting/progressively_build_global_model.py \
                                                -w -o outputs/global_model/kmeans-200-2000 \
                                                -m outputs/rubble-pixsfm_local_models  \
                                                -i client_image_lists/rubble-pixsfm_kmeans-15 \
                                                -data datasets/rubble-pixsfm \
                                                --sh-degree 3
```

## Evaluation
```bash
python gaussian-splatting/eval.py -w -o eval/kmeans-10-20_000 -g outputs/global_model/kmeans-10-20_000/global_model_epoch20000.pth -data datasets/rubble-pixsfm
```