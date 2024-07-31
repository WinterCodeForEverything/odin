set -e

export DETECTRON2_DATASETS="/projects/katefgroup/language_grounding/SEMSEG_100k"
SCANNET_DATA_DIR="/path/to/train_validation_database.yaml"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 python train_odin.py  --dist-url='tcp://127.0.0.1:8474' --num-gpus 2 --config-file configs/scannet_context/swin_3d.yaml \
OUTPUT_DIR /projects/katefgroup/language_grounding/bdetr2/arxiv_reproduce/scannet_swinb_benchmark SOLVER.IMS_PER_BATCH 4 \
SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 4000 \
INPUT.FRAME_LEFT 9 INPUT.FRAME_RIGHT 9 INPUT.SAMPLING_FRAME_NUM 19 \
MODEL.WEIGHTS '/projects/katefgroup/language_grounding/odin_arxiv/m2f_coco_swin.pkl' \
SOLVER.BASE_LR 1e-4 \
INPUT.IMAGE_SIZE 256 \
MODEL.CROSS_VIEW_CONTEXTUALIZE True \
INPUT.CAMERA_DROP True \
INPUT.STRONG_AUGS True \
INPUT.AUGMENT_3D True \
INPUT.VOXELIZE True \
INPUT.SAMPLE_CHUNK_AUG True \
MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
MODEL.CROSS_VIEW_BACKBONE True \
DATASETS.TRAIN "('scannet_context_instance_trainval_20cls_single_100k',)" \
DATASETS.TEST "('scannet_context_instance_test_20cls_single_highres_100k',)" \
MODEL.PIXEL_DECODER_PANET True \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
SKIP_CLASSES "[19, 20]" \
USE_GHOST_POINTS True \
MODEL.FREEZE_BACKBONE False \
SOLVER.TEST_IMS_PER_BATCH 2 \
SAMPLING_STRATEGY "consecutive" \
USE_SEGMENTS True \
SOLVER.MAX_ITER 100000 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.TEST_NUM_WORKERS 2 \
MAX_FRAME_NUM 250 \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
USE_WANDB True \
USE_MLP_POSITIONAL_ENCODING True \
SCANNET_DATA_DIR $SCANNET_DATA_DIR
# EXPORT_BENCHMARK_DATA True \
# EXPORT_BENCHMARK_PATH "/projects/katefgroup/language_grounding/benchmark_evaluations/odin_arxiv_benchmark_scannet" \
# SCANNET_DATA_DIR '/projects/katefgroup/language_grounding/mask3d_processed/scannet/test_database.yaml' \

# reduce lr at 99k iterations to 1e-5, get the benchmark submission checkpoint at 10.5k \
# DATASETS.TEST "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k',)" \
# and MAX_FRAME_NUM 250 (because some test scenes have large number of frames and result in OOM on a 48G gpu)


# Use the following flags for exporting the results
# EXPORT_BENCHMARK_DATA True \
# EXPORT_BENCHMARK_PATH "/projects/katefgroup/language_grounding/benchmark_evaluations/odin_arxiv_benchmark_scannet"
# and also change DATASETS.TEST to "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k',)"
# and add SCANNET_DATA_DIR '/projects/katefgroup/language_grounding/mask3d_processed/scannet/test_database.yaml'

