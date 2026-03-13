import sys
from musetalk.utils.face_detection import FaceAlignment, LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

# Lazy-loaded models (initialized on first use)
_model = None
_fa = None

def _ensure_models_loaded():
    """Lazily initialize DWPose and FaceAlignment models on first use."""
    global _model, _fa
    if _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
        _model = init_model(config_file, checkpoint_file, device=device)
    if _fa is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

# marker if the bbox is not sufficient
coord_placeholder = (0.0,0.0,0.0,0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"Unable to read image: {img_path}")
        frames.append(frame)
    return frames

def _detect_faces(img_list, upperbondrange=0):
    """
    Shared logic for face detection and landmark extraction.
    Returns (coords_list, frames, average_range_minus, average_range_plus).
    """
    _ensure_models_loaded()
    if not img_list:
        return [], [], [], []

    frames = read_imgs(img_list)
    coords_list = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for frame in tqdm(frames):
        try:
            results = inference_topdown(_model, frame)
            if not results:
                coords_list.append(coord_placeholder)
                continue
            results = merge_data_samples(results)
            keypoints = getattr(results.pred_instances, "keypoints", None)
            if keypoints is None or len(keypoints) == 0 or keypoints.shape[1] < 91:
                coords_list.append(coord_placeholder)
                continue
            face_land_mark = keypoints[0][23:91].astype(np.int32)
            bbox = _fa.get_detections_for_batch(np.asarray([frame]))
            face_bbox = bbox[0] if bbox else None
        except Exception as exc:
            print(f"face detection failed for one frame: {exc}")
            coords_list.append(coord_placeholder)
            continue

        if face_bbox is None:  # no face in the image
            coords_list.append(coord_placeholder)
            continue

        half_face_coord = face_land_mark[29].copy()
        range_minus = (face_land_mark[30] - face_land_mark[29])[1]
        range_plus = (face_land_mark[29] - face_land_mark[28])[1]
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)
        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        min_upper_bond = 0
        upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)

        f_landmark = (
            int(np.min(face_land_mark[:, 0])),
            int(upper_bond),
            int(np.max(face_land_mark[:, 0])),
            int(np.max(face_land_mark[:, 1])),
        )
        x1, y1, x2, y2 = f_landmark

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            coords_list.append(face_bbox)
            print("error bbox:", face_bbox)
        else:
            coords_list.append(f_landmark)

    return coords_list, frames, average_range_minus, average_range_plus

def _format_range_text(num_frames, average_range_minus, average_range_plus, upperbondrange):
    """Format the bbox shift range information text."""
    if len(average_range_minus) == 0 or len(average_range_plus) == 0:
        return f"Total frame:「{num_frames}」 No faces detected, cannot compute adjustment range."
    avg_minus = int(sum(average_range_minus) / len(average_range_minus))
    avg_plus = int(sum(average_range_plus) / len(average_range_plus))
    return f"Total frame:「{num_frames}」 Manually adjust range : [ -{avg_minus}~{avg_plus} ] , the current value: {upperbondrange}"

def get_bbox_range(img_list, upperbondrange=0):
    coords_list, frames, average_range_minus, average_range_plus = _detect_faces(img_list, upperbondrange)
    return _format_range_text(len(frames), average_range_minus, average_range_plus, upperbondrange)

def get_landmark_and_bbox(img_list, upperbondrange=0):
    coords_list, frames, average_range_minus, average_range_plus = _detect_faces(img_list, upperbondrange)
    text_range = _format_range_text(len(frames), average_range_minus, average_range_plus, upperbondrange)
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(text_range)
    print("*************************************************************************************************************************************")
    return coords_list, frames


if __name__ == "__main__":
    # pickle usage is required for compatibility with existing coord cache files
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)

    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)

    print(coords_list)
