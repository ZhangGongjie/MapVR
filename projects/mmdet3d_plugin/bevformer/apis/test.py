# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time
import json

import tqdm
import cv2
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util

from tools.lane_evaluator import LaneEvaluator, line_classes, polygon_classes


WIDTH = 240
HEIGHT = 480
PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
INST_THRESHOLD_LINE = 0.05
INST_THRESHOLD_POLYGON = 0.05
IOUTHRS_LINE = np.linspace(0.25, 0.50, int(np.round((0.50 - 0.25) / 0.05)) + 1)
IOUTHRS_POLYGON = np.linspace(0.50, 0.75, int(np.round((0.75 - 0.50) / 0.05)) + 1)
MAXDETS_LINE = [1, 10, 100]
MAXDETS_POLYGON = [1, 10, 100]
QUERY_LINE = [
    ("AP", "all", "all", 100),
    ("AP", "all", "divider", 100),
    ("AP", "all", "boundary", 100),
    ("AP", 0.25, "all", 100),
    ("AP", 0.25, "divider", 100),
    ("AP", 0.25, "boundary", 100),
    ("AP", 0.50, "all", 100),
    ("AP", 0.50, "divider", 100),
    ("AP", 0.50, "boundary", 100),
    ("AR", "all", "all", 1),
    ("AR", "all", "all", 10),
    ("AR", "all", "all", 100),
    ("AR", "all", "divider", 100),
    ("AR", "all", "boundary", 100),
]
QUERY_POLYGON = [
    ("AP", "all", "ped_crossing", 100),
    ("AP", 0.50, "ped_crossing", 100),
    ("AP", 0.75, "ped_crossing", 100),
    ("AR", "all", "ped_crossing", 1),
    ("AR", "all", "ped_crossing", 10),
    ("AR", "all", "ped_crossing", 100),
]
DILATION_LINE = 5
DILATION_POLYGON = 5


def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]


def post_process_instance_polygon(data, preds):
    def instance_to_segm_polygon(params):
        normalized_params = np.copy(params)
        normalized_params[:, 0] = (normalized_params[:, 0] - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * WIDTH
        normalized_params[:, 1] = (normalized_params[:, 1] - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * HEIGHT
        points = np.around(normalized_params).astype(int)
        mask = np.zeros([HEIGHT, WIDTH])
        cv2.fillPoly(mask, [points], 1)
        return (mask > 0).astype(float)

    def instance_to_segm_line(params):
        normalized_params = np.copy(params)
        normalized_params[:, 0] = (normalized_params[:, 0] - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * WIDTH
        normalized_params[:, 1] = (normalized_params[:, 1] - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * HEIGHT
        points = np.around(normalized_params).astype(int)
        mask = np.zeros([HEIGHT, WIDTH])
        cv2.polylines(mask, [points], False, 1, 1)
        return (mask > 0).astype(float)

    gts = []
    dts = []

    gt_vectors = data["vectors"]
    for vector in gt_vectors:
        if vector["type"] in polygon_classes.values():
            gts.append({
                "class": vector["type"],
                "segmentation": instance_to_segm_polygon(vector["pts"]),
            })

    pred_instances = preds["pts_bbox"]
    for score, label, pts in zip(pred_instances["scores_3d"], pred_instances["labels_3d"], pred_instances["pts_3d"]):
        if score < INST_THRESHOLD_POLYGON or len(pts) < 2:
            continue
        if label in polygon_classes.values():
            dts.append({
                "class": label,
                "segmentation": instance_to_segm_polygon(pts),
                "score": score,
            })

    return gts, dts


def post_process_instance_line(data, preds):
    def instance_to_segm_line(params):
        normalized_params = np.copy(params)
        normalized_params[:, 0] = (normalized_params[:, 0] - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * WIDTH
        normalized_params[:, 1] = (normalized_params[:, 1] - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * HEIGHT
        points = np.around(normalized_params).astype(int)
        mask = np.zeros([HEIGHT, WIDTH])
        cv2.polylines(mask, [points], False, 1, 1)
        return (mask > 0).astype(float)

    gts = []
    dts = []

    gt_vectors = data["vectors"]
    for vector in gt_vectors:
        if vector["type"] in line_classes.values():
            gts.append({
                "class": vector["type"],
                "segmentation": instance_to_segm_line(vector["pts"]),
            })

    pred_instances = preds["pts_bbox"]
    for score, label, pts in zip(pred_instances["scores_3d"], pred_instances["labels_3d"], pred_instances["pts_3d"]):
        if score < INST_THRESHOLD_LINE or len(pts) < 2:
            continue
        if label in line_classes.values():
            dts.append({
                "class": label,
                "segmentation": instance_to_segm_line(pts),
                "score": score,
            })

    return gts, dts


def custom_single_gpu_test(model, data_loader):
    model.eval()
    bbox_results = []
    mask_results = []
    coco_results_line = []
    coco_results_polygon = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    have_mask = False

    with open(dataset.map_ann_file, "r") as f:
        gt_data = json.load(f)["GTs"]
    assert len(gt_data) == len(dataset)
    gt_data_dict = {}
    for gt in gt_data:
        gt_data_dict[gt["sample_token"]] = gt

    evaluator_line = LaneEvaluator(classes=line_classes,
                                   parameterization="instanceseg",
                                   post_process_func=post_process_instance_line,
                                   iouThrs=IOUTHRS_LINE,
                                   maxDets=MAXDETS_LINE,
                                   query=QUERY_LINE,
                                   dilation=DILATION_LINE,
                                   to_thin=False,
                                   width=WIDTH,
                                   height=HEIGHT)
    evaluator_polygon = LaneEvaluator(classes=polygon_classes,
                                      parameterization="instanceseg",
                                      post_process_func=post_process_instance_polygon,
                                      iouThrs=IOUTHRS_POLYGON,
                                      maxDets=MAXDETS_POLYGON,
                                      query=QUERY_POLYGON,
                                      dilation=DILATION_POLYGON,
                                      to_thin=False,
                                      width=WIDTH,
                                      height=HEIGHT)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            token = data["img_metas"][0].data[0][0]["sample_idx"]
            coco_results_line.append(evaluator_line.evaluate(gt_data_dict[token], result[0]))
            coco_results_polygon.append(evaluator_polygon.evaluate(gt_data_dict[token], result[0]))

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        for _ in range(batch_size):
            prog_bar.update()

    if not have_mask:
        return bbox_results, [{"type": "line", "evaluator": evaluator_line, "results": coco_results_line}, {"type": "polygon", "evaluator": evaluator_polygon, "results": coco_results_polygon}]
    return {'bbox_results': bbox_results, 'mask_results': mask_results}, [{"type": "line", "evaluator": evaluator_line, "results": coco_results_line}, {"type": "polygon", "evaluator": evaluator_polygon, "results": coco_results_polygon}]


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    coco_results_line = []
    coco_results_polygon = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False

    with open(dataset.map_ann_file, "r") as f:
        gt_data = json.load(f)["GTs"]
    assert len(gt_data) == len(dataset)
    gt_data_dict = {}
    for gt in gt_data:
        gt_data_dict[gt["sample_token"]] = gt

    evaluator_line = LaneEvaluator(classes=line_classes,
                                   parameterization="instanceseg",
                                   post_process_func=post_process_instance_line,
                                   iouThrs=IOUTHRS_LINE,
                                   maxDets=MAXDETS_LINE,
                                   query=QUERY_LINE,
                                   dilation=DILATION_LINE,
                                   to_thin=False,
                                   width=WIDTH,
                                   height=HEIGHT)
    evaluator_polygon = LaneEvaluator(classes=polygon_classes,
                                      parameterization="instanceseg",
                                      post_process_func=post_process_instance_polygon,
                                      iouThrs=IOUTHRS_POLYGON,
                                      maxDets=MAXDETS_POLYGON,
                                      query=QUERY_POLYGON,
                                      dilation=DILATION_POLYGON,
                                      to_thin=False,
                                      width=WIDTH,
                                      height=HEIGHT)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            token = data["img_metas"][0].data[0][0]["sample_idx"]
            coco_results_line.append(evaluator_line.evaluate(gt_data_dict[token], result[0]))
            coco_results_polygon.append(evaluator_polygon.evaluate(gt_data_dict[token], result[0]))

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
        coco_results_line = collect_results_gpu(coco_results_line, len(dataset))
        coco_results_polygon = collect_results_gpu(coco_results_polygon, len(dataset))
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir_mask = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir_mask)
        else:
            mask_results = None
        tmpdir_coco_line = tmpdir + "_coco_line" if tmpdir is not None else None
        tmpdir_coco_polygon = tmpdir + "_coco_polygon" if tmpdir is not None else None
        coco_results_line = collect_results_cpu(coco_results_line, len(dataset), tmpdir_coco_line)
        coco_results_polygon = collect_results_cpu(coco_results_polygon, len(dataset), tmpdir_coco_polygon)

    if mask_results is None:
        return bbox_results, [{"type": "line", "evaluator": evaluator_line, "results": coco_results_line}, {"type": "polygon", "evaluator": evaluator_polygon, "results": coco_results_polygon}]
    return {'bbox_results': bbox_results, 'mask_results': mask_results}, [{"type": "line", "evaluator": evaluator_line, "results": coco_results_line}, {"type": "polygon", "evaluator": evaluator_polygon, "results": coco_results_polygon}]


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
