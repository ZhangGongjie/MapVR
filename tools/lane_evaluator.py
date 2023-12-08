import os
import json
import datetime
from functools import reduce

import cv2
import tqdm
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from pycocotools import mask
from skimage.morphology import dilation, square, thin
import pdb


line_classes = {"divider": 0, "boundary": 2}
polygon_classes = {"ped_crossing": 1}


def compute_iou(gts, dts, height, width, dilation_size, to_thin):
    dilation_size = int(dilation_size)

    if len(gts) == 0 or len(dts) == 0:
        return [], 0.0

    frame_gt = np.zeros([height, width])
    frame_dt = np.zeros([height, width])

    masks_gt = []
    masks_dt = []

    for i in range(len(gts)):
        assert gts[i]["segmentation"].shape == (height, width)
        instance_gt = thin(gts[i]["segmentation"]) if to_thin else gts[i]["segmentation"]
        if dilation_size > 0:
            instance_gt = dilation(instance_gt, square(dilation_size))
        frame_gt += instance_gt
        masks_gt.append(mask.encode(np.asfortranarray(instance_gt.astype(np.uint8))))
    for i in range(len(dts)):
        assert dts[i]["segmentation"].shape == (height, width)
        instance_dt = thin(dts[i]["segmentation"]) if to_thin else dts[i]["segmentation"]
        if dilation_size > 0:
            instance_dt = dilation(instance_dt, square(dilation_size))
        frame_dt += instance_dt
        masks_dt.append(mask.encode(np.asfortranarray(instance_dt.astype(np.uint8))))

    ious = mask.iou(masks_dt, masks_gt, np.zeros([len(masks_gt)]))

    # frame-level mIoU
    frame_gt = (frame_gt > 0).astype(np.uint8)
    frame_dt = (frame_dt > 0).astype(np.uint8)
    frame_iou = mask.iou(
        [mask.encode(np.asfortranarray(frame_dt))],
        [mask.encode(np.asfortranarray(frame_gt))],
        np.array([0])
    ).item()

    return ious, frame_iou


class FrameIOUeval:

    def __init__(self, classes, height=640, width=320, dilation=5, to_thin=False):
        self.height = height
        self.width = width
        self.dilation = dilation
        self.classes = classes
        self.to_thin = to_thin

        self.catIds = [v for k, v in classes.items()]

    def evaluate(self, gts, dts):
        """
        gts:
            class: scalar
            segmentation: [H, W]
        dts:
            class: scalar
            segmentation: [H, W]
        """
        result = {}
        for catId in self.catIds:
            result[catId] = self._evaluateImg(gts, dts, catId)
        return result

    def _evaluateImg(self, gts, dts, catId):
        gt = [gt for gt in gts if gt["class"] == catId]
        dt = [dt for dt in dts if dt["class"] == catId]
        assert len(gt) == 1  # one and only one ground truth map
        assert len(dt) <= 1  # at most one detection map
        if len(dt) == 0:
            return 0.0

        _, frame_iou = compute_iou(gt, dt, self.height, self.width, self.dilation, self.to_thin)
        return frame_iou

    def summarize(self, frame_ious):
        results = dict()
        for catStr, catId in self.classes.items():
            cat_frame_ious = [frame[catId] for frame in frame_ious]
            frame_iou = np.mean(cat_frame_ious) if len(cat_frame_ious) > 0 else -1
            print("Frame mIoU [{}] = {:.3f}".format(catStr, frame_iou))
            results[f"Frame mIoU,{catStr}"] = frame_iou

        return results


class LaneCOCOeval:

    def __init__(self, classes, height=640, width=320, dilation=5, to_thin=False):
        self.height = height
        self.width = width
        self.dilation = dilation
        self.classes = classes
        self.to_thin = to_thin

        self.params = Params()

        self.catIds = [v for k, v in classes.items()]

    def evaluate(self, gts, dts):
        """
        gts:
            class: scalar
            segmentation: [H, W]
        dts:
            class: scalar
            segmentation: [H, W]
            score: scalar
        """
        result = {}
        for catId in self.catIds:
            result[catId] = self._evaluateImg(gts, dts, catId, max(self.params.maxDets))
        return result

    def _evaluateImg(self, gts, dts, catId, maxDet):
        gt_indices = [index for index in range(len(gts)) if gts[index]["class"] == catId]
        dt_indices = [index for index in range(len(dts)) if dts[index]["class"] == catId]
        gt = [gts[index] for index in gt_indices]
        dt = [dts[index] for index in dt_indices]

        if len(gt) == 0 and len(dt) == 0:
            return None

        # sort detections by score
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt_indices = [dt_indices[i] for i in dtind[:maxDet]]
        dt = [dt[i] for i in dtind[:maxDet]]

        ious, frame_iou = compute_iou(gt, dt, self.height, self.width, self.dilation, self.to_thin)

        # matching
        T = len(self.params.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G)) - 1  # -1 means unmatched
        dtm = np.zeros((T, D)) - 1  # -1 means unmatched
        if len(ious) > 0:
            for tind, t in enumerate(self.params.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m = -1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt is already matched, continue
                        if gtm[tind, gind] >= 0:
                            continue
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        iou = ious[dind, gind]
                        m = gind
                    if m == -1:
                        continue
                    # if match made, store id of match for both dt and gt
                    dtm[tind, dind] = m
                    gtm[tind, m] = dind

        return {
            "category_id": catId,
            "maxDet": maxDet,
            "dtIds": dt_indices,
            "gtIds": gt_indices,
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "frame_iou": frame_iou,
            "ious": ious,
        }

    def _accumulate(self, evalImgs):
        T = len(self.params.iouThrs)
        R = len(self.params.recThrs)
        K = len(self.catIds)
        M = len(self.params.maxDets)
        precision = -np.ones((T, R, K, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, M))
        scores = -np.ones((T, R, K, M))

        for k in range(len(self.catIds)):
            for m, maxDet in enumerate(self.params.maxDets):
                E = [frame[self.catIds[k]] for frame in evalImgs]
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e["dtScores"][:maxDet] for e in E])

                inds = np.argsort(-dtScores, kind="mergesort")
                dtScoresSorted = dtScores[inds]

                dtm = np.concatenate([e["dtMatches"][:, :maxDet] for e in E], axis=1)[:, inds]
                gtm = np.concatenate([e["gtMatches"] for e in E], axis=1)
                npig = gtm.shape[1]
                if npig == 0:
                    continue

                tps = dtm >= 0
                fps = dtm < 0

                tp_sum = np.cumsum(tps, axis=1).astype(np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp + tp + np.spacing(1))
                    q = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t, k, m] = rc[-1]
                    else:
                        recall[t, k, m] = 0

                    pr = pr.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    inds = np.searchsorted(rc, self.params.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t, :, k, m] = np.array(q)
                    scores[t, :, k, m] = np.array(ss)

        self.eval = {
            "params": self.params,
            "counts": [T, R, K, M],
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }

    def summarize(self, evalImgs):
        def _summarize(ap=1, iouThr=None, catStr='all', maxDet=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | class={:>12s} | maxDet={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDet]
            if ap == 1:
                # dimension of precision: [TxRxKxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if catStr != 'all':
                    t = [self.catIds.index(self.classes[catStr])]
                    s = s[:, :, t, :]
                s = s[:, :, :, mind]
            else:
                # dimension of recall: [TxKxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if catStr != 'all':
                    t = [self.catIds.index(self.classes[catStr])]
                    s = s[:, t, :]
                s = s[:, :, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, catStr, maxDet, mean_s))
            return mean_s

        self._accumulate(evalImgs)

        results = dict()

        for query in self.params.query:
            query_type, iouThr, catStr, maxDet = query

            assert isinstance(query_type, str)
            assert query_type.upper() in ["AP", "AR"]

            if isinstance(iouThr, str):
                assert iouThr == "all"
                iouThr = None
                str_iouThr = "{:0.2f}:{:0.2f}".format(self.params.iouThrs[0], self.params.iouThrs[-1])
            elif isinstance(iouThr, float):
                assert iouThr in self.params.iouThrs
                str_iouThr = "{:0.2f}".format(iouThr)
            else:
                assert False

            assert isinstance(catStr, str)
            assert catStr in self.classes or catStr == "all"

            assert maxDet in self.params.maxDets

            res = _summarize(1 if query_type.upper() == "AP" else 0, iouThr=iouThr, catStr=catStr, maxDet=maxDet)
            results[",".join([query_type.upper(), str_iouThr, catStr, str(maxDet)])] = res

        for catStr, catId in self.classes.items():
            cat_frame_ious = [frame[catId]["frame_iou"] for frame in evalImgs if frame[catId] is not None]
            frame_iou = np.mean(cat_frame_ious) if len(cat_frame_ious) > 0 else -1
            print("Frame mIoU [{}] = {:.3f}".format(catStr, frame_iou))
            results[f"Frame mIoU,{catStr}"] = frame_iou

        return results


class Params:
    def __init__(self):
        self.iouThrs = np.linspace(0.20, 0.60, int(np.round((0.60 - 0.20) / 0.05)) + 1)
        self.recThrs = np.linspace(0.00, 1.00, int(np.round((1.00 - 0.00) / 0.01)) + 1)
        self.maxDets = [1, 10, 100]

        """
        query format
            type: AP or AR
            iouThr: "all" or value in iouThrs
            class: "all" or value in classes
            maxDet: value in maxDets
        """
        self.query = [
            ("AP", "all", "all", 100),
            ("AP", "all", "lane", 100),
            ("AP", "all", "curbside", 100),
            ("AP", "all", "stopline", 100),
            ("AP", 0.20, "all", 100),
            ("AP", 0.20, "lane", 100),
            ("AP", 0.20, "curbside", 100),
            ("AP", 0.20, "stopline", 100),
            ("AP", 0.40, "all", 100),
            ("AP", 0.40, "lane", 100),
            ("AP", 0.40, "curbside", 100),
            ("AP", 0.40, "stopline", 100),
            ("AP", 0.60, "all", 100),
            ("AP", 0.60, "lane", 100),
            ("AP", 0.60, "curbside", 100),
            ("AP", 0.60, "stopline", 100),
            ("AR", "all", "all", 1),
            ("AR", "all", "all", 10),
            ("AR", "all", "all", 100),
            ("AR", "all", "lane", 100),
            ("AR", "all", "curbside", 100),
            ("AR", "all", "stopline", 100),
        ]


class LaneEvaluator:
    def __init__(self,
                 classes,
                 parameterization,
                 post_process_func,
                 iouThrs=None,
                 maxDets=None,
                 query=None,
                 width=320,
                 height=640,
                 dilation=5,
                 to_thin=False):

        self.parameterization = parameterization
        self.classes = classes

        if parameterization == "instanceseg":
            evaluator = LaneCOCOeval(classes=classes, height=height, width=width, dilation=dilation, to_thin=to_thin)
            if iouThrs is not None:
                evaluator.params.iouThrs = iouThrs
            if maxDets is not None:
                evaluator.params.maxDets = maxDets
            if query is not None:
                evaluator.params.query = query
        elif parameterization == "semanticseg":
            evaluator = FrameIOUeval(height=height, width=width, dilation=dilation, to_thin=to_thin)
        else:
            raise NotImplementedError("Unsupported parameterization: {}".format(parameterization))

        self.evaluator = evaluator

        self.width = width
        self.height = height
        self.post_process_func = post_process_func

    def evaluate(self, data, preds):
        gts, dts = self.post_process_func(data, preds)
        return self.evaluator.evaluate(gts, dts)

    def summarize(self, per_frame_results):
        return self.evaluator.summarize(per_frame_results)


def demo_post_process_instanceseg(data, preds):
    """
    read out ground truth and detections (convert to numpy arrays)
    post-process detections (if needed)
    generate sample lists with following keys
    gts:
        class: scalar
        segmentation: [H, W]; binarized float
    dts:
        class: scalar
        segmentation: [H, W]; binarized float
        score: scalar
    """
    gts = []
    dts = []

    lane_inst_threshold = 0.4
    lane_inst_seg_threshold = 0.3

    gt_lane = data["gt_lane"]
    target_classes = gt_lane["lane_classes"].squeeze(0).cpu().numpy()  # [50]
    target_inst_segmentations = gt_lane["inst_seg"].squeeze(0).cpu().numpy()  # [50, H, W]
    num_lanes = gt_lane["n_lanes"]  # [1]

    for i in range(num_lanes):
        gts.append({
            "class": target_classes[i],
            "segmentation": target_inst_segmentations[i],
        })

    pred_classes = preds["all_lane_cls_preds"][-1].sigmoid().squeeze(0).cpu().numpy()  # [50, 4]
    pred_inst_segmentations = preds["all_lane_inst_seg_preds"][-1].sigmoid().double().squeeze(0).cpu().numpy()  # [50, H, W]

    for class_name, class_idx in line_classes.items():
        pred_lanes_idx = np.where(pred_classes[:, class_idx] > lane_inst_threshold)[0]
        num_pred = len(pred_lanes_idx)
        scores = pred_classes[pred_lanes_idx, class_idx]
        pred_inst_seg_masks = pred_inst_segmentations[pred_lanes_idx]

        for i in range(num_pred):
            dts.append({
                "class": class_idx,
                "segmentation": (pred_inst_seg_masks[i] > lane_inst_seg_threshold).astype(float),
                "score": scores[i],
            })

    return gts, dts


def demo_post_process_instance(data, preds):
    """
    Example of converting lane instances (20-point representation) to segmentation maps

    read out ground truth and detections (convert to numpy arrays)
    post-process detections (if needed)
    generate sample lists with following keys
    gts:
        class: scalar
        segmentation: [H, W]; binarized float
    dts:
        class: scalar
        segmentation: [H, W]; binarized float
        score: scalar
    """
    height = 640
    width = 320

    def instance_to_segm(params):
        """
        convert params [20, 2] to segmentation map [H, W]
        """
        points = np.around((params * np.array([width, height]))).astype(int)
        mask = np.zeros([height, width])
        cv2.polylines(mask, [points], False, 1, 1)
        return (mask > 0).astype(float)

    gts = []
    dts = []

    lane_inst_threshold = 0.4

    gt_lane = data["gt_lane"]
    target_classes = gt_lane["lane_classes"].squeeze(0).cpu().numpy()  # [50]
    target_params = gt_lane["params"].squeeze(0).cpu().numpy()  # [50, 20, 2]
    num_lanes = gt_lane["n_lanes"]  # [1]

    for i in range(num_lanes):
        gts.append({
            "class": target_classes[i],
            "segmentation": instance_to_segm(target_params[i]),
        })

    pred_classes = preds["all_lane_cls_preds"][-1].softmax().squeeze(0).cpu().numpy()  # [50, 4]
    pred_params = preds["all_lane_reg_preds"][-1].double().squeeze(0).unflatten(-1, (20, 2)).cpu().numpy()  # [50, 20, 2]

    for class_name, class_idx in line_classes.items():
        pred_lanes_idx = np.where(pred_classes[:, class_idx] > lane_inst_threshold)[0]
        num_pred = len(pred_lanes_idx)
        scores = pred_classes[pred_lanes_idx, class_idx]
        pred_points = pred_params[pred_lanes_idx]

        for i in range(num_pred):
            dts.append({
                "class": class_idx,
                "segmentation": instance_to_segm(pred_points[i]),
                "score": scores[i],
            })

    return gts, dts


def demo_post_process_semanticseg(data, preds):
    """
    read out ground truth and detections (convert to numpy arrays)
    post-process detections (if needed)
    generate sample lists with following keys
    gts:
        class: scalar
        segmentation: [H, W]; binarized float
    dts:
        class: scalar
        segmentation: [H, W]; binarized float
    """
    gts = []
    dts = []

    semanticseg_thres = 0.2

    lane_seg_gt = (data["gt_lane"].squeeze(0).cpu().numpy() > 0).astype(float)  # [H, W, 3]
    lane_seg_dt = (preds["lane_preds"].sigmoid().squeeze(0).cpu().numpy() > semanticseg_thres).astype(float)  # [H, W, 3]

    for class_name, class_idx in line_classes.items():
        gts.append({
            "class": class_idx,
            "segmentation": lane_seg_gt[:, :, class_idx - 1],
        })
        dts.append({
            "class": class_idx,
            "segmentation": lane_seg_dt[:, :, class_idx - 1],
        })

    return gts, dts
