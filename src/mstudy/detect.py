"""Person detection and body-pose estimation.

Both models are Apache-2.0 and run on ONNX Runtime, so there is no PyTorch in
this project and nothing here is AGPL-encumbered. That matters: the obvious
alternative (Ultralytics YOLO) is AGPL-3.0, which would restrict what this
system could become later.
"""

from __future__ import annotations

import numpy as np
from rtmlib import RTMPose, YOLOX
from rtmlib.tools.object_detection.post_processings import multiclass_nms
from rtmlib.tools.solution.body import Body

from .config import Config

# COCO-17 is what RTMPose returns.
N_KEYPOINTS = 17


class ScoredYOLOX(YOLOX):
    """YOLOX that also returns its confidence scores.

    rtmlib's YOLOX computes per-detection scores and then discards them (its own
    docstring says "NOT returned: final_scores"). ByteTrack needs them - its
    whole method is to associate high-confidence detections first and then
    recover objects from the low-confidence ones. Without scores we would be
    throwing away the part of the tracker that survives occlusion, which in a
    workshop full of steel stock is exactly the part we need.
    """

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_p, ratio = self.preprocess(image)
        outputs = self.inference(image_p)[0]

        if outputs.shape[-1] == 4 or outputs.shape[-1] > 5:
            strides = [8, 16, 32]
            hsizes = [self.model_input_size[0] // s for s in strides]
            wsizes = [self.model_input_size[1] // s for s in strides]

            grids, expanded = [], []
            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                expanded.append(np.full((*grid.shape[:2], 1), stride))

            grids = np.concatenate(grids, 1)
            expanded = np.concatenate(expanded, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded

            preds = outputs[0]
            boxes, scores = preds[:, :4], preds[:, 4:5] * preds[:, 5:]

            xyxy = np.ones_like(boxes)
            xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            xyxy /= ratio

            dets, _ = multiclass_nms(xyxy, scores, self.nms_thr, self.score_thr)
            if dets is None or len(dets) == 0:
                return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
            # dets columns: x1, y1, x2, y2, score, class
            return dets[:, :4].astype(np.float32), dets[:, 4].astype(np.float32)

        # ONNX variant with NMS baked in: (x1, y1, x2, y2, score)
        boxes, scores = outputs[0, :, :4] / ratio, outputs[0, :, 4]
        keep = scores > self.score_thr
        return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


class PersonAnalyser:
    """Detector + pose estimator, configured from a Config."""

    def __init__(self, cfg: Config):
        preset = Body.MODE[cfg.detection.mode]
        self.detector = ScoredYOLOX(
            preset["det"],
            model_input_size=preset["det_input_size"],
            score_thr=cfg.detection.min_confidence,
            backend=cfg.detection.backend,
            device=cfg.detection.device,
        )
        self.pose_model = RTMPose(
            preset["pose"],
            model_input_size=preset["pose_input_size"],
            backend=cfg.detection.backend,
            device=cfg.detection.device,
        )

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (boxes as (N,4) xyxy, scores as (N,))."""
        return self.detector(image)

    def pose(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (keypoints (N,17,2), keypoint_scores (N,17))."""
        if len(boxes) == 0:
            return np.zeros((0, N_KEYPOINTS, 2), np.float32), np.zeros((0, N_KEYPOINTS), np.float32)
        kpts, scores = self.pose_model(image, bboxes=boxes)
        return np.asarray(kpts, np.float32), np.asarray(scores, np.float32)


def empty_pose(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Placeholder pose for frames where we deliberately skipped estimation.

    Zero confidence tells `compute_motion` to ignore these rows rather than
    treat a missing wrist as a wrist that did not move.
    """
    return (
        np.zeros((n, N_KEYPOINTS, 2), np.float32),
        np.zeros((n, N_KEYPOINTS), np.float32),
    )
