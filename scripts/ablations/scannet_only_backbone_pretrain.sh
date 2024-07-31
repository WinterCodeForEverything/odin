set -e

export DETECTRON2_DATASETS="/projects/katefgroup/language_grounding/SEMSEG_100k"
SCANNET_DATA_DIR="/path/to/train_validation_database.yaml"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 python train_odin.py  --dist-url='tcp://127.0.0.1:1921' --num-gpus 2  --resume --config-file configs/scannet_context/3d.yaml \
OUTPUT_DIR /projects/katefgroup/language_grounding/bdetr2/arxiv_reproduce/ablation_scannet_only_resnet_pretrain SOLVER.IMS_PER_BATCH 6 \
SOLVER.CHECKPOINT_PERIOD 4000 TEST.EVAL_PERIOD 4000 \
INPUT.FRAME_LEFT 12 INPUT.FRAME_RIGHT 12 INPUT.SAMPLING_FRAME_NUM 25 \
MODEL.WEIGHTS 'detectron2://ImageNetPretrained/torchvision/R-50.pkl' \
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
DATASETS.TRAIN "('scannet_context_instance_train_20cls_single_highres_100k',)" \
DATASETS.TEST "('scannet_context_instance_val_20cls_single_highres_100k','scannet_context_instance_train_eval_20cls_single_highres_100k',)" \
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
MAX_FRAME_NUM -1 \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
USE_WANDB True \
USE_MLP_POSITIONAL_ENCODING True \
SCANNET_DATA_DIR $SCANNET_DATA_DIR

# MODEL.WEIGHTS 'detectron2://ImageNetPretrained/torchvision/R-50.pkl' \
# reduce lr at 32k iterations to 1e-5, get the best checkpoint at 1.5k iterations
