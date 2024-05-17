
## Logs

- 2024.05.17-0 init 
- 2024.05.17-1 使用Kmeans聚15类划分，并保证所有图片全覆盖，大约相邻范围内有10-40左右的重叠视角
- 2024.05.17-2 单机双卡训练


## Training Local Models:
```bash
python tools/gen_client_data.py -d datasets/rubble-pixsfm \
                                -o datasets/rubble-pixsfm_image_lists \
                                --n-clients 15 --n-data-max 150 
```



``` bash
bash scripts/client_training.sh 0 20 outputs/rubble-pixsfm_colmap_results \
                                     datasets/rubble-pixsfm \
                                     datasets/rubble-pixsfm_image_lists \
                                     outputs/rubble-pixsfm_local_models
```

``` bash
python gaussian-splatting/train.py -s outputs/rubble-pixsfm_colmap_results/00004 \
                                   -i datasets/rubble-pixsfm/train/rgbs \
                                   -w \
                                   -m outputs/rubble-pixsfm_local_models/00004
```

``` bash
python gaussian-splatting/build_global_model.py \
                                                -w -o outputs/global_model \
                                                -m outputs/rubble-pixsfm_local_models  \
                                                -i datasets/rubble-pixsfm_image_lists \
                                                -data datasets/rubble-pixsfm \
                                                --sh-degree 3
```

