import argparse
import copy
import glob
import json
import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending, get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import datagen, load_all_model


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def video2imgs(vid_path, save_path, ext=".png", cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
        count += 1
    cap.release()


def osmakedirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)


def sorted_image_paths(image_dir):
    image_paths = glob.glob(os.path.join(image_dir, "*.[jpJP][pnPN]*[gG]"))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem.lower())

    return sorted(image_paths, key=sort_key)


class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:
            self.base_path = f"./results/avatars/{avatar_id}"

        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": args.version,
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()

    def _load_existing_avatar(self):
        self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location="cpu")
        with open(self.coords_path, "rb") as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = sorted_image_paths(self.full_imgs_path)
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, "rb") as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = sorted_image_paths(self.mask_out_path)
        self.mask_list_cycle = read_imgs(input_mask_list)

    def _recreate_avatar(self):
        if os.path.exists(self.avatar_path):
            shutil.rmtree(self.avatar_path)
        print("*********************************")
        print(f"  creating avator: {self.avatar_id}")
        print("*********************************")
        osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
        self.prepare_material()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                should_recreate = getattr(args, "force_recreate", False)
                if not should_recreate and sys.stdin.isatty():
                    response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                    should_recreate = response.lower() == "y"

                if should_recreate:
                    self._recreate_avatar()
                else:
                    self._load_existing_avatar()
            else:
                self._recreate_avatar()
            return

        if not os.path.exists(self.avatar_path):
            raise RuntimeError(
                f"{self.avatar_id} does not exist, you should set preparation to True"
            )

        with open(self.avatar_info_path, "r") as f:
            avatar_info = json.load(f)

        if avatar_info["bbox_shift"] != self.avatar_info["bbox_shift"]:
            should_recreate = getattr(args, "force_recreate", False)
            if not should_recreate and sys.stdin.isatty():
                response = input("bbox_shift changed, you need to re-create it. Continue? (c/continue)")
                should_recreate = response.lower() == "c"
            else:
                should_recreate = True

            if should_recreate:
                self._recreate_avatar()
            else:
                raise RuntimeError("User declined to recreate avatar with changed bbox_shift.")
        else:
            self._load_existing_avatar()

    @torch.no_grad()
    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext=".png")
        else:
            print(f"copy files in {self.video_path}")
            for src_path in sorted_image_paths(self.video_path):
                filename = os.path.basename(src_path)
                shutil.copyfile(src_path, os.path.join(self.full_imgs_path, filename))

        input_img_list = sorted_image_paths(self.full_imgs_path)
        if not input_img_list:
            raise ValueError(f"No input frames were found for avatar source: {self.video_path}")

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        valid_frames = []
        valid_coords = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, frame.shape[1])
            if args.version == "v15":
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
            bbox = [x1, y1, x2, y2]
            if x2 <= x1 or y2 <= y1:
                continue
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
            valid_frames.append(frame)
            valid_coords.append(bbox)

        if not input_latent_list:
            raise ValueError(f"No valid face was detected in avatar source: {self.video_path}")

        self.frame_list_cycle = valid_frames + valid_frames[::-1]
        self.coord_list_cycle = valid_coords + valid_coords[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for cached_img in sorted_image_paths(self.full_imgs_path):
            os.remove(cached_img)
        for cached_mask in sorted_image_paths(self.mask_out_path):
            os.remove(cached_mask)

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = args.parsing_mode if args.version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, "wb") as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, "wb") as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, self.latents_out_path)

    def process_frames(self, res_frame_queue, skip_save_images):
        while True:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            if res_frame is None:
                break

            bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, ori_frame.shape[1])
            y2 = min(y2, ori_frame.shape[0])
            bbox = [x1, y1, x2, y2]
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                self.idx += 1
                continue

            mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if not skip_save_images:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx += 1

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        tmp_dir = os.path.join(self.avatar_path, "tmp")
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        if not skip_save_images:
            os.makedirs(tmp_dir, exist_ok=True)

        print("start inference")
        start_time = time.time()
        audio_result = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
        if audio_result is None:
            raise FileNotFoundError(f"Unable to load audio features from: {audio_path}")
        whisper_input_features, librosa_length = audio_result
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")

        video_num = len(whisper_chunks)
        if video_num == 0:
            raise ValueError(f"No whisper chunks were generated for audio: {audio_path}")

        res_frame_queue = queue.Queue()
        self.idx = 0
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size, device=device)
        start_time = time.time()

        try:
            for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size))):
                audio_feature_batch = pe(whisper_batch.to(device))
                latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

                pred_latents = unet.model(
                    latent_batch,
                    timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample
                pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_queue.put(res_frame)
        finally:
            res_frame_queue.put(None)
            process_thread.join()

        if skip_save_images:
            print(f"Total process time of {video_num} frames without saving images = {time.time() - start_time}s")
        else:
            print(f"Total process time of {video_num} frames including saving images = {time.time() - start_time}s")

        if out_vid_name is not None and not skip_save_images:
            if self.idx == 0:
                raise RuntimeError("Realtime inference did not generate any output frames.")

            subprocess.run([
                "ffmpeg", "-y", "-v", "warning", "-r", str(fps),
                "-f", "image2", "-i", f"{self.avatar_path}/tmp/%08d.png",
                "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18",
                f"{self.avatar_path}/temp.mp4"
            ], check=True)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            subprocess.run([
                "ffmpeg", "-y", "-v", "warning",
                "-i", f"{self.avatar_path}/temp.mp4", "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                output_vid
            ], check=True)

            os.remove(f"{self.avatar_path}/temp.mp4")
            shutil.rmtree(tmp_dir)
            print(f"result is save to {output_vid}")
        elif os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default="./results", help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help="Use saved coordinates to save time")
    parser.add_argument("--saved_coord", action="store_true", help="Save coordinates for future use")
    parser.add_argument("--parsing_mode", default="jaw", help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--skip_save_images", action="store_true", help="Whether skip saving images for better generation speed calculation")
    parser.add_argument("--force_recreate", action="store_true", help="Force recreate avatar without interactive prompt")

    args = parser.parse_args()

    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        path_separator = ";" if sys.platform == "win32" else ":"
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    if device.type == "cuda":
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
    else:
        pe = pe.to(device)
        vae.vae = vae.vae.to(device)
        unet.model = unet.model.to(device)

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
        )
    else:
        fp = FaceParsing()

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        if args.version == "v15":
            bbox_shift = 0
        else:
            bbox_shift = inference_config[avatar_id]["bbox_shift"]
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=args.batch_size,
            preparation=data_preparation,
        )

        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print("Inferring using:", audio_path)
            avatar.inference(audio_path, audio_num, args.fps, args.skip_save_images)
