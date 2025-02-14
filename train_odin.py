# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import warnings
warnings.filterwarnings('ignore')

import copy
import itertools
import logging
import os
import gc
import weakref
import time

from collections import OrderedDict
from typing import Any, Dict, List, Set

import numpy as np
import torch
from torch import nn

from scipy.spatial import ConvexHull

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    AMPTrainer,
    SimpleTrainer
)
from detectron2.evaluation import (
    DatasetEvaluator,
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from odin.data_video.dataset_mapper_coco import COCOInstanceNewBaselineDatasetMapper

from odin import (
    ScannetDatasetMapper,
    Scannet3DEvaluator,
    ScannetSemantic3DEvaluator,
    COCOEvaluatorMemoryEfficient,
    add_maskformer2_video_config,
    add_maskformer2_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
    build_detection_train_loader_multi_task,
)

from odin.data_video.build import merge_datasets
from odin.global_vars import SCANNET_LIKE_DATASET
from torchinfo import summary

torch.multiprocessing.set_sharing_strategy('file_system')

import ipdb
st = ipdb.set_trace

from contextlib import ExitStack, contextmanager

from tqdm import tqdm


class OneCycleLr_D2(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}
        

def create_ddp_model(model, *, fp16_compression=False, find_unused_parameters=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs, find_unused_parameters=find_unused_parameters)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        # super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=cfg.MULTI_TASK_TRAINING or cfg.FIND_UNUSED_PARAMETERS)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(
        cls, cfg, dataset_name,
        output_folder=None, use_2d_evaluators_only=False,
        use_3d_evaluators_only=False,
    ):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluators = []
        if cfg.TEST.EVAL_3D and cfg.MODEL.DECODER_3D and not use_2d_evaluators_only:
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                evaluators.append(
                        ScannetSemantic3DEvaluator(
                            dataset_name, 
                            output_dir=output_folder, 
                            eval_sparse=cfg.TEST.EVAL_SPARSE,
                            cfg=cfg
                        ))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                evaluators.append(
                        Scannet3DEvaluator(
                            dataset_name,
                            output_dir=output_folder,
                            eval_sparse=cfg.TEST.EVAL_SPARSE,
                            cfg=cfg
                        ))
        if (cfg.TEST.EVAL_2D or cfg.EVAL_PER_IMAGE) and not use_3d_evaluators_only:
            if cfg.INPUT.ORIGINAL_EVAL:
                print("Using original COCO Eval, potentially is RAM hungry")
                evaluators.append(COCOEvaluator(dataset_name, output_dir=output_folder, use_fast_impl=False))
            else:
                evaluators.append(COCOEvaluatorMemoryEfficient(
                    dataset_name, output_dir=output_folder, use_fast_impl=False,
                    per_image_eval=cfg.EVAL_PER_IMAGE, evaluate_subset=cfg.EVALUATE_SUBSET,))
        return evaluators

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MULTI_TASK_TRAINING:
            if cfg.TRAIN_3D:
                if len(cfg.DATASETS.TRAIN_3D) > 1:
                    dataset_dicts = [get_detection_dataset_dicts(
                        cfg.DATASETS.TRAIN_3D[i],
                        proposal_files=None,
                    ) for i in range(len(cfg.DATASETS.TRAIN_3D))]
                    mappers = [
                        ScannetDatasetMapper(cfg, is_train=True, dataset_name=dataset_name, dataset_dict=dataset_dict) for dataset_name, dataset_dict in zip(cfg.DATASETS.TRAIN, dataset_dicts)
                    ]
                    dataset_dict_3d = merge_datasets(dataset_dicts, mappers, balance=cfg.BALANCE_3D_DATASETS)
                    mapper_3d = None
                else:
                    dataset_dict_3d = get_detection_dataset_dicts(
                            cfg.DATASETS.TRAIN_3D,
                            proposal_files=None,
                    )
                    mapper_3d = ScannetDatasetMapper(
                        cfg, is_train=True,
                        dataset_name=cfg.DATASETS.TRAIN_3D[0],
                        dataset_dict=dataset_dict_3d
                    )
            else:
                dataset_dict_3d = None
                mapper_3d = None
            
            if cfg.TRAIN_2D:
                    dataset_dict_2d = get_detection_dataset_dicts(
                        cfg.DATASETS.TRAIN_2D,
                        proposal_files=None,
                    )
                    if 'coco' in cfg.DATASETS.TRAIN_2D[0]:
                        mapper_2d = COCOInstanceNewBaselineDatasetMapper(cfg, True, dataset_name=cfg.DATASETS.TRAIN_2D[0])
                    else:
                        mapper_2d = ScannetDatasetMapper(
                            cfg, is_train=True,
                            dataset_name=cfg.DATASETS.TRAIN_2D[0],
                            dataset_dict=dataset_dict_2d,
                            force_decoder_2d=cfg.FORCE_DECODER_3D,
                            frame_left=0,
                            frame_right=0,
                            decoder_3d=False
                        )
            else:
                dataset_dict_2d = None
                mapper_2d = None
            
            return build_detection_train_loader_multi_task(
                    cfg, mapper_3d=mapper_3d, mapper_2d=mapper_2d,
                    dataset_3d=dataset_dict_3d, dataset_2d=dataset_dict_2d
            )
        else:
            dataset_name = cfg.DATASETS.TRAIN[0]
            scannet_like = False
            for scannet_like_dataset in SCANNET_LIKE_DATASET:
                if scannet_like_dataset in dataset_name:
                    scannet_like = True
                    break

            if scannet_like:
                dataset_dict = get_detection_dataset_dicts(
                    dataset_name,
                    proposal_files=None,
                )
                mapper = ScannetDatasetMapper(cfg, is_train=True, dataset_name=dataset_name, dataset_dict=dataset_dict)
                return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)
            elif 'coco' in dataset_name:
                mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True, dataset_name=dataset_name)
                return build_detection_train_loader(cfg, mapper=mapper)
            else:
                raise NotImplementedError

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        scannet_like = False
        for scannet_like_dataset in SCANNET_LIKE_DATASET:
            if scannet_like_dataset in dataset_name:
                scannet_like = True
                break
        if scannet_like:
            dataset_dict = get_detection_dataset_dicts(
                [dataset_name],
                proposal_files=[
                    cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
                ]
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subsample_data=cfg.TEST.SUBSAMPLE_DATA if dataset_name in cfg.DATASETS.TEST_SUBSAMPLED else None,
            )
            mapper = ScannetDatasetMapper(
                cfg, is_train=False, dataset_name=dataset_name, dataset_dict=dataset_dict,
                decoder_3d=False if dataset_name in cfg.DATASETS.TEST_2D_ONLY else cfg.MODEL.DECODER_3D,
            )
            return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)
        elif 'coco' in dataset_name:
            dataset_dict = get_detection_dataset_dicts(
                [dataset_name],
                proposal_files=[
                    cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
                ]
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subsample_data=cfg.TEST.SUBSAMPLE_DATA if dataset_name in cfg.DATASETS.TEST_SUBSAMPLED else None,
            )
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, is_train=False, dataset_name=dataset_name)
            return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)
        else:
            raise NotImplementedError

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        if cfg.SOLVER.LR_SCHEDULER_NAME == "onecyclelr":
            return OneCycleLr_D2(
                optimizer,
                max_lr=cfg.SOLVER.BASE_LR,
                total_steps=cfg.SOLVER.MAX_ITER,
            )
        else:
            return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        print(summary(model))

        panet_resnet_layers = ['cross_view_attn', 'res_to_trans', 'trans_to_res']
        panet_swin_layers = ['cross_view_attn', 'cross_layer_norm', 'res_to_trans', 'trans_to_res']

        if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
            backbone_panet_layers = panet_resnet_layers
        elif cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
            backbone_panet_layers = panet_swin_layers
        else:
            raise NotImplementedError


        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name :
                    # panet layers are initialize from scratch so use default lr
                    panet_found = False
                    for panet_name in backbone_panet_layers:
                        if panet_name in module_name:
                            hyperparams["lr"] = hyperparams["lr"]
                            panet_found = True
                            break

                    if not panet_found:
                        hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(
                        cfg, dataset_name, use_2d_evaluators_only=dataset_name in cfg.DATASETS.TEST_2D_ONLY if cfg.MULTI_TASK_TRAINING else False,
                        use_3d_evaluators_only=dataset_name in cfg.DATASETS.TEST_3D_ONLY if cfg.MULTI_TASK_TRAINING else False,)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

            
        gc.collect()
        torch.cuda.empty_cache()
        
        if not cfg.MULTI_TASK_TRAINING:
            #format for writer
            if len(results) == 1:
                results_structured = list(results.values())[0]
            elif len(results) == 2:
                # find a better way than hard-coding here
                results_val = results[cfg.DATASETS.TEST[0]].copy()
                suffix = '_full' if 'single' in cfg.DATASETS.TEST[0] else ''
                suffix += f'_{dataset_name.split("_")[0]}'
                results_val = {f'val{suffix}'+k: v for k, v in results_val.items()}
                # st()
                try:
                    if cfg.EVAL_PER_IMAGE:
                        results_val[f'val_{dataset_name.split("_")[0]}segm'] = results_val[f'val{suffix}segm']
                        del results_val[f'val{suffix}segm']
                except:
                    print("Error in Logging")
                    print(results_val.keys(), print(f'val{suffix}segm'))
                results_train = results[cfg.DATASETS.TEST[1]].copy()
                results_train = {f'train{suffix}'+k: v for k, v in results_train.items()}
                try:
                    if cfg.EVAL_PER_IMAGE:
                        results_train[f'train_{dataset_name.split("_")[0]}segm'] = results_train[f'train{suffix}segm']
                        del results_train[f'train{suffix}segm']
                except:
                    print(results_train.keys(), print(f'train{suffix}segm'))
                results_structured = {}
                results_structured.update(results_train)
                results_structured.update(results_val)

            else:
                for dataset_name in cfg.DATASETS.TEST:
                    results_structured = {}
                    suffix = 'train_full' if 'train_eval' in dataset_name else 'val_full'
                    results_val = results[dataset_name].copy()
                    results_val = {f'{suffix}_{dataset_name.split("_")[0]}'+k: v for k, v in results_val.items()}
                    results_structured.update(results_val)

        else:
            results_structured = {}
            for dataset_name in cfg.DATASETS.TEST_3D_ONLY:
                if dataset_name in results:
                    suffix = 'train_full' if 'train_eval' in dataset_name else 'val_full'
                    suffix += f'_{dataset_name.split("_")[0]}'
                    results_val = results[dataset_name].copy()
                    results_val = {f'{suffix}'+k: v for k, v in results_val.items()}
                    results_structured.update(results_val)
                
            for dataset_name in cfg.DATASETS.TEST_2D_ONLY:
                if dataset_name in results:
                    suffix = 'train' if 'train_eval' in dataset_name else 'val'
                    suffix += f'_{dataset_name.split("_")[0]}'
                    results_val = results[dataset_name].copy()
                    results_val = {f'{suffix}'+k: v for k, v in results_val.items()}
                    results_structured.update(results_val)
        return results_structured

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        self._trainer.iter = self.iter
        
        assert self._trainer.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast
        
        assert self.cfg.SOLVER.AMP.ENABLED

        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast(dtype=self._trainer.precision):
            loss_dict = self._trainer.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                loss_custom = None
                if 'loss_3d' in loss_dict or 'loss_2d' in loss_dict:
                    loss_name = 'loss_3d' if 'loss_3d' in loss_dict else 'loss_2d'
                    loss_custom = loss_dict[loss_name]
                    loss_dict.pop('loss_3d', None)
                    loss_dict.pop('loss_2d', None)
                losses = sum(loss_dict.values())
                
                if loss_custom is not None:
                    loss_dict[loss_name] = loss_custom

        self._trainer.optimizer.zero_grad()
        self._trainer.grad_scaler.scale(losses).backward()
        
        self._trainer.after_backward()

        self._trainer._write_metrics(loss_dict, data_time)

        self._trainer.grad_scaler.step(self.optimizer)
        self._trainer.grad_scaler.update()  

    @classmethod
    def save_output(cls, cfg, model, path = None, align_file_path = None):
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        @contextmanager
        def inference_context(model):
            """
            A context where the model is temporarily changed to eval mode,
            and restored to previous mode afterwards.

            Args:
                model: a torch Module
            """
            training_mode = model.training
            model.eval()
            yield
            model.train(training_mode)
        
        #all_feats = {}
        for _, dataset_name in enumerate(cfg.DATASETS.TEST):

            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())
            
            data_loader = cls.build_test_loader(cfg, dataset_name)
            #data_loader = cls.build_train_loader(cfg)
            #print("Start inference on {} batches".format(len(data_loader)))
            #output = []
            #all_feats = {}
            for  data in tqdm(data_loader):
                #print(data[0].keys())
                #print_dict(data[0])
                #print('depth_file_names', len(data[0]['depth_file_names']) )
                #print(data[0]['file_name'])
                scene_id = data[0]['image_id']
                scannet_coords = data[0]['scannet_coords'] 
                intrinsics = torch.stack(data[0]['intrinsics'] )
                image_shape = ( 256, 320 )
                poses = torch.stack(data[0]['poses'])
                # images = data[0]['images']
                # all_classes = data[0]['all_classes']
                # image_file_names = data[0]['file_names']
                # align_matrix = np.eye(4)
                # with open(os.path.join(align_file_path, scene_id, '%s.txt'%(scene_id)), 'r') as f:
                #     for line in f:
                #         if line.startswith('axisAlignment'):
                #             align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
                #             break
                # pts = torch.ones((scannet_coords.shape[0], 4), dtype=scannet_coords.dtype)
                # pts[:, 0:3] = scannet_coords
                
                # align_coords = torch.matmul(pts, torch.tensor(align_matrix, device=pts.device, dtype=pts.dtype).T)[:, :3]
                # #align_coords = np.dot(pts, align_matrix.transpose())[:, :3]  # Nx4
                # assert (torch.sum(torch.isnan(align_coords)) == 0)
                
                #scannet_color = data[0]['scannet_color'].cpu().numpy()
                
                              
                model.eval()
                scene_feats = {}
                with torch.no_grad():
                    results, feature = model(data)
                    results_i = results[0]
                    #print_dict(results_i)
                    #print_dict(feature)
                    pred_ins_masks = results_i['instances_3d']['pred_masks'].cpu()
                    ins_num = pred_ins_masks.shape[1]
                    #ins_cat = results_i['instances_3d']['pred_classes'].cpu().numpy()
                    
                    # gt_ins_masks = results_i['instances_3d']['scannet_gt_masks'].cpu()
                    # gt_ins_cat = results_i['instances_3d']['scannet_gt_classes'].cpu().numpy()
                    # gt_ins_num = gt_ins_masks.shape[0]
                    #print(pred_ins_id.shape)
                    #pred_ins_id = pred_ins_masks.argmax(1).numpy()
                    #print(pred_ins_id[:200])
                    
                    #pred_sem_id = results_i['semantic_3d'].cpu().numpy()
                    #print(pred_sem_id[:20])
                    

                    
                    img_features = {}
                    for k, v in feature.items():
                        img_features[k] = v.permute(0, 2, 3, 1)
                    
                    
                    for i in range(ins_num):
                        ins_pc = scannet_coords[pred_ins_masks[:, i].bool()]
                        if ins_pc.shape[0] == 0:
                            continue
                        
                        
                        ins_img_masks = map_3d_points_to_2d(ins_pc, intrinsics, poses, image_shape)
                        
                        # for b in range(ins_img_masks.shape[0]):
                        #     if ins_img_masks[b].sum() < 100:
                        #         continue
                        #     visualize_projection(b, ins_img_masks[b], images[b], output_dir=f'{path}/{scene_id}/{i:02}_{all_classes[ins_cat[i]-1]}' ) 
                        
                        # if i > 20:
                        #     break
                        # continue

                        ins_feat = {}
                        for k, feats in img_features.items():
                            scale = ins_img_masks.shape[1] // feats.shape[1]
                            ins_feat_mask = max_pooling_masks(ins_img_masks, scale)
                            # print(ins_img_masks.shape)
                            # print(ins_feat_mask.shape)
                            # print(feats.shape)
                            assert ins_feat_mask.shape == feats.shape[:3]
                            ins_multiview_feats = []
                            for b in range(ins_feat_mask.shape[0]):
                                feat = feats[b].view(-1, feats.shape[-1])
                                mask = ins_feat_mask[b].view(-1)
                                if mask.sum() == 0:
                                    continue
                                # print(feat[mask].shape)
                                # print( feats[b][ins_feat_mask[b]].shape)
                                # return
                                ins_multiview_feat = feat[mask].mean(0)
                                ins_multiview_feats.append( {"volume": mask.sum(), 
                                                             "feat": ins_multiview_feat.cpu().detach()
                                                             })
                                
                            ins_multiview_feats.sort(key=lambda x: x["volume"], reverse=True)
                            ins_feat[k] = ins_multiview_feats[:4] if len(ins_multiview_feats) > 4 else ins_multiview_feats
                            #print_dict(ins_feat)
                            # only keep the top 4 views
                        #print(f"{scene_id}: {len(all_feats)}/{ins_num}")
                        
                        scene_feats[f"{scene_id}_{i:02}"] = ins_feat
                torch.save(scene_feats, f'{path}/{scene_id}.pt')
                print(f"{scene_id}: {len(scene_feats)}/{ins_num}")
                    
        #torch.save(all_feats, path)
        #print(f"{scene_id}: {len(all_feats)}/{ins_num}")
                    #torch.save(( scannet_coords, scannet_color, pred_sem_id, pred_ins_id, img_feature ), f'{path}/{scene_id}.pth')
            #results[dataset_name] = output
                #break


def print_dict(d, n = 0):
    if isinstance(d, dict):
        for k, v in d.items():
            print('  '*n, k)
            print_dict(v, n+1 )
    elif isinstance(d, torch.Tensor):
        print('  '*n, d.shape)
    elif isinstance(d, list):
        for i in d:
            print_dict(i, n+1)
    #elif isinstance(d, int):
    else:
        print('  '*n, d)

def compute_3d_bounding_boxes(points: torch.Tensor, instance_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D bounding boxes for multiple instances.
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing 3D point coordinates.
        instance_mask (torch.Tensor): Boolean tensor of shape (N, I) where I is the number
                                        of instances. True indicates the point belongs to the instance.
    
    Returns:
        torch.Tensor: A tensor of shape (I, 6) where each row is [x_min, y_min, z_min, x_max, y_max, z_max]
                      for the corresponding instance.
    """
    # Number of instances (I)
    num_instances = instance_mask.shape[1]
    bboxes = []

    for i in range(num_instances):
        # Get the mask for the current instance: shape (N,)
        mask = instance_mask[:, i]
        # Check if any points belong to this instance
        if mask.sum() == 0:
            # If no points, we can set a default box (or you might choose to skip/raise an error)
            bbox = torch.zeros(6, dtype=points.dtype, device=points.device)
        else:
            # Select the points that belong to the current instance
            pts = points[mask]  # shape (n_i, 3) where n_i is the number of points in instance i
            # Compute min and max along each axis (dim=0)
            min_vals, _ = pts.min(dim=0)
            max_vals, _ = pts.max(dim=0)
            size = max_vals - min_vals
            center = (max_vals + min_vals) / 2
            # Concatenate the minimum and maximum coordinates to form the bounding box
            bbox = torch.cat([min_vals, max_vals], dim=0)  # shape (6,)
        bboxes.append(bbox)

    # Stack all bounding boxes into a tensor of shape (I, 6)
    return torch.stack(bboxes, dim=0)


def visualize_projection(b, shot_mask, image, output_dir="output_images"):
    """
    Visualize the projection of 3D points onto a 2D image and save the output images.
    Args:
        b (int): Batch index.
        shot_mask (torch.Tensor): Tensor of shape (H, W) containing the mask.
        image (torch.Tensor): Tensor of shape (3, H, W) containing the image.
        output_dir (str): Directory to save the output images.
    """
    import matplotlib.pyplot as plt

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the image tensor to a numpy array and transpose to (H, W, 3)
    image_np = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    mask = shot_mask.cpu().numpy()
    plt.imshow(mask, alpha=0.5, cmap='Reds')
    plt.title(f"Projection on Image {b}")

    output_path = os.path.join(output_dir, f"projection_{b}.jpg")
    plt.savefig(output_path)
    plt.close()
    
    

        
def map_3d_points_to_2d(points, intrinsics, poses, image_size):
    """
    Map 3D points to 2D image coordinates.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing 3D point coordinates.
        intrinsics (torch.Tensor): Tensor of shape (B, 3, 3) containing camera intrinsics.
        poses (torch.Tensor): Tensor of shape (B, 4, 4) containing camera poses.
        image_size (tuple): Tuple (H, W) representing the image size.

    Returns:
        torch.Tensor: Tensor of shape (B, H, W) containing the shot mask.
    """
    from skimage.draw import polygon
    
    N = points.shape[0]
    B, _, _ = intrinsics.shape
    H, W = image_size

    # # Convert points to homogeneous coordinates
    # points_h = torch.cat([points, torch.ones(N, 1, device=points.device)], dim=1)  # (N, 4)

    shot_masks = torch.zeros((B, H, W), dtype=torch.bool, device=points.device)

    # poses_inv = torch.inverse(poses)
    points = points.unsqueeze(0).expand(B, -1, -1)
    points_2d, valid_mask = project(intrinsics, poses, points, image_size)
    
    for i in range(B):
        
        points_2d_i = points_2d[i][valid_mask[i]]

        if points_2d_i.shape[0] < 3:
            continue

        # Compute the convex hull of the projected points
        try:
            hull = ConvexHull(points_2d_i.cpu().numpy())
        except:
            continue
        
        hull_points = points_2d_i[hull.vertices]

        # Create a mask for the convex hull
        rr, cc = polygon(hull_points[:, 1].cpu().numpy(), hull_points[:, 0].cpu().numpy(), (H, W))
        shot_masks[i, rr, cc] = True

    return shot_masks

def project(intrinsics, poses, world_coords, image_size):
    """
    Projects 3D world coordinates back to 2D image coordinates and filters out invalid points.

    Inputs:
        intrinsics: B X 3 X 3 (camera intrinsics)
        poses: B X 4 X 4 (camera extrinsics)
        world_coords: B X N X 3 (3D world coordinates)
        image_size: tuple (H, W) representing the image dimensions

    Outputs:
        valid_pixel_coords: List of tensors of shape (N_valid, 2) containing only valid 2D pixel coordinates
        valid_mask: List of tensors of shape (N_valid,) indicating valid points
    """
    B, N, _ = world_coords.shape
    img_H, img_W = image_size

    # Convert 3D world coordinates to homogeneous form (B, N, 4)
    ones = torch.ones((B, N, 1), device=world_coords.device)
    world_coords_h = torch.cat([world_coords, ones], dim=-1)  # B X N X 4

    # Transform world coordinates to camera coordinates (B X N X 4)
    cam_coords_h = torch.matmul(torch.inverse(poses), world_coords_h.transpose(1, 2)).transpose(1, 2)  # B X N X 4

    # Convert to non-homogeneous 3D camera coordinates
    cam_coords = cam_coords_h[..., :3] / cam_coords_h[..., 3:4]  # B X N X 3

    # Extract intrinsics parameters
    fx = intrinsics[:, 0, 0].unsqueeze(1)  # B X 1
    fy = intrinsics[:, 1, 1].unsqueeze(1)  # B X 1
    px = intrinsics[:, 0, 2].unsqueeze(1)  # B X 1
    py = intrinsics[:, 1, 2].unsqueeze(1)  # B X 1

    # Compute 2D pixel coordinates
    x = (cam_coords[..., 0] * fx / cam_coords[..., 2]) + px
    y = (cam_coords[..., 1] * fy / cam_coords[..., 2]) + py

    # Stack pixel coordinates
    pixel_coords = torch.stack([x, y], dim=-1)  # B X N X 2

    # Validity mask: check if points are in front of the camera and within image bounds
    valid = (cam_coords[..., 2] > 0) & (x >= 0) & (x < img_W) & (y >= 0) & (y < img_H)

    # # Collect only valid points per batch
    # valid_pixel_coords = []
    # valid_mask = []

    # for b in range(B):
    #     valid_points = pixel_coords[b][valid[b]]  # Extract valid 2D points
    #     valid_pixel_coords.append(valid_points)
    #     valid_mask.append(valid[b])  # Flatten validity mask

    return pixel_coords, valid

# def project(intrinsics, poses, world_coords):
#     """
#     Projects 3D world coordinates back to 2D image coordinates.

#     Inputs:
#         intrinsics: B X V X 3 X 3 (camera intrinsics)
#         poses: B X V X 4 X 4 (camera extrinsics)
#         world_coords: B X V X H X W X 3 (3D world coordinates)

#     Outputs:
#         pixel_coords: B X V X H X W X 2 (2D pixel coordinates in image space)
#         valid: B X V X H X W (bool indicating valid points)
#     """
#     B, V, H, W, _ = world_coords.shape

#     # Convert world coordinates to homogeneous coordinates
#     ones = torch.ones_like(world_coords[..., :1])
#     world_coords_h = torch.cat([world_coords, ones], dim=-1)  # B X V X H X W X 4

#     # Transform world coordinates to camera coordinates
#     cam_coords_h = torch.matmul(torch.inverse(poses), world_coords_h.unsqueeze(-1))  # B X V X H X W X 4 X 1
#     cam_coords_h = cam_coords_h.squeeze(-1)  # B X V X H X W X 4

#     # Convert to 3D camera coordinates (remove homogeneous scale)
#     cam_coords = cam_coords_h[..., :3] / cam_coords_h[..., 3:4]  # B X V X H X W X 3

#     # Extract intrinsics
#     fx = intrinsics[..., 0, 0][..., None, None, None]  # B X V X 1 X 1 X 1
#     fy = intrinsics[..., 1, 1][..., None, None, None]
#     px = intrinsics[..., 0, 2][..., None, None, None]
#     py = intrinsics[..., 1, 2][..., None, None, None]

#     # Compute 2D pixel coordinates
#     x = (cam_coords[..., 0] * fx / cam_coords[..., 2]) + px
#     y = (cam_coords[..., 1] * fy / cam_coords[..., 2]) + py

#     # Stack to get final pixel coordinates
#     pixel_coords = torch.stack([x, y], dim=-1)  # B X V X H X W X 2

#     # Validity mask: check if points are in front of the camera
#     valid = cam_coords[..., 2] > 0  # Depth must be positive

#     return pixel_coords, valid




def max_pooling_masks(masks: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Apply max pooling to masks and ensure the output mask type is bool.

    Args:
        masks (torch.Tensor): Tensor of shape (B, H, W) containing masks.
        scale (int): Pooling scale, e.g., 2, 4, 8, 16.

    Returns:
        torch.Tensor: Tensor of shape (B, H//scale, W//scale) containing pooled masks.
    """
    pooled_masks = nn.functional.max_pool2d(masks.unsqueeze(1).float(), kernel_size=scale).squeeze(1)
    return pooled_masks.bool()



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="odin")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        return res
    elif args.save_output:
        #print("################# Saving Output #################")
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        out_dir = '/mnt/ssd/liuchao/odin/odin_3d_ins_seg2'
        res = Trainer.save_output(cfg, model, 
                                  path= out_dir,  #'/mnt/ssd/liuchao/odin/odin_3d_ins_seg2', 
                                  align_file_path='/mnt/ssd/liuchao/ScanNet/scans')  
        all_feats = {}
        for filename in os.listdir(out_dir):
            if filename.endswith('.pt'):
                all_feats.update(torch.load(os.path.join(out_dir, filename), map_location='cpu'))
        torch.save(all_feats, os.path.join( "/home/liuchao/Chat-Scene/annotations/scannet_odin_videofeats.pt"))

        return res     

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    paser = default_argument_parser()
    paser.add_argument("--save_output", action="store_true")
    args = paser.parse_args()
    print("Command Line Args:", args)
    
    # this is needed to prevent memory leak in conv2d layers
    # see: https://github.com/pytorch/pytorch/issues/98688#issuecomment-1869290827
    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1' 
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
