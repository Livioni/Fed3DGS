
$$ Logs

- 2024.05.17 init 


$$ Training Local Models:

``` bash
bash scripts/client_training.sh 0 20 outputs/rubble-pixsfm_colmap_results 
                                     datasets/rubble-pixsfm 
                                     datasets/rubble-pixsfm_image_lists 
                                     outputs/rubble-pixsfm_local_models
```

``` bash
python gaussian-splatting/train.py -s outputs/rubble-pixsfm_colmap_results/00004 
                                   -i datasets/rubble-pixsfm/train/rgbs 
                                   -w 
                                   -m outputs/rubble-pixsfm_local_models/00004
```

``` bash
python gaussian-splatting/build_global_model.py 
                                                -w -o outputs/global_model 
                                                -m outputs/rubble-pixsfm_local_models  
                                                -i datasets/rubble-pixsfm_image_lists 
                                                -data datasets/rubble-pixsfm
                                                --sh-degree 3
```

