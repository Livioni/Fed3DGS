# 设置CUDA环境变量以仅使用GPU 0
export CUDA_VISIBLE_DEVICES=1

# 从命令行参数读取目录和参数
COLMAP_RESULTS_DIR=$3
DATASET_ROOT=$4
IMAGE_LIST_DIR=$5
OUTPUT_DIR=$6

# 循环处理每个序列
for i in `seq -f '%05g' $1 $2`; do
    python gaussian-splatting/train.py -s $COLMAP_RESULTS_DIR/$i -i $DATASET_ROOT/train/rgbs -w -m $OUTPUT_DIR/$i --port 6011
done