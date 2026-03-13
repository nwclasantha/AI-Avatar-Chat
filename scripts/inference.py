import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys
import shlex

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def sorted_image_paths(image_dir):
    image_paths = glob.glob(os.path.join(image_dir, '*.[jpJP][pnPN]*[gG]'))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem.lower())

    return sorted(image_paths, key=sort_key)

@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
    
    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path, 
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    # Convert models to half precision if float16 is enabled
    use_float16 = args.use_float16 and device.type == "cuda"
    if args.use_float16 and device.type != "cuda":
        print("float16 inference requires CUDA; falling back to float32.")
    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
        
    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()
    
    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)
    
    # Process each task
    for task_id in inference_config:
        try:
            # Get task configuration
            video_path = inference_config[task_id]["video_path"]
            audio_path = inference_config[task_id]["audio_path"]
            task_output_vid_name = inference_config[task_id].get("result_name", args.output_vid_name)
            
            # Set bbox_shift based on version
            if args.version == "v15":
                bbox_shift = 0  # v15 uses fixed bbox_shift
            else:
                bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)  # v1 uses config or default
            
            # Set output paths
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}_{audio_basename}"
            
            # Create temporary directories
            temp_dir = os.path.join(args.result_dir, f"{args.version}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Set result save paths
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
            if os.path.isdir(result_img_save_path):
                shutil.rmtree(result_img_save_path)
            os.makedirs(result_img_save_path, exist_ok=True)
            
            # Set output video paths
            if task_output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, task_output_vid_name)
            output_vid_name_concat = os.path.join(temp_dir, output_basename + "_concat.mp4")
            
            # Extract frames from source video
            save_dir_full = None  # only set for video inputs
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(temp_dir, input_basename)
                if os.path.isdir(save_dir_full):
                    shutil.rmtree(save_dir_full)
                os.makedirs(save_dir_full, exist_ok=True)
                subprocess.run([
                    "ffmpeg", "-v", "fatal", "-i", video_path,
                    "-start_number", "0", f"{save_dir_full}/%08d.png"
                ], check=True)
                input_img_list = sorted_image_paths(save_dir_full)
                fps = get_video_fps(video_path)
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = sorted_image_paths(video_path)
                fps = args.fps
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
            if not input_img_list:
                raise ValueError(f"No input frames were found for: {video_path}")
            fps = max(1, int(round(fps)))

            # Extract audio features
            audio_result = audio_processor.get_audio_feature(audio_path)
            if audio_result is None:
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
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
            if len(whisper_chunks) == 0:
                raise ValueError(f"No whisper chunks were generated for audio: {audio_path}")
            
            # Preprocess input images
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                print("Using saved coordinates")
                with open(crop_coord_save_path, 'rb') as f:
                    coord_list = pickle.load(f)
                frame_list = read_imgs(input_img_list)
                if len(coord_list) != len(frame_list):
                    print("Saved coordinates do not match frame count, recomputing landmarks")
                    coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                    with open(crop_coord_save_path, 'wb') as f:
                        pickle.dump(coord_list, f)
            else:
                print("Extracting landmarks... time-consuming operation")
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)
            
            print(f"Number of frames: {len(frame_list)}")         
            
            # Process each frame
            input_latent_list = []
            valid_frame_indices = []
            for frame_idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, frame.shape[1])
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
                valid_frame_indices.append(frame_idx)

            if not input_latent_list:
                raise ValueError(f"No valid face was detected in: {video_path}")
        
            # Smooth first and last frames
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            valid_frame_indices_cycle = valid_frame_indices + valid_frame_indices[::-1]
            
            # Batch inference
            print("Starting inference")
            video_num = len(whisper_chunks)
            batch_size = args.batch_size
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )
            
            res_frame_list = []
            total = int(np.ceil(float(video_num) / batch_size))
            
            # Execute inference
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                audio_feature_batch = pe(whisper_batch)
                latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
                
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            
            # Pad generated images to original video size
            print("Padding generated images to original video size")
            write_idx = 0
            for i, res_frame in enumerate(tqdm(res_frame_list)):
                source_idx = valid_frame_indices_cycle[i % len(valid_frame_indices_cycle)]
                bbox = coord_list[source_idx]
                ori_frame = copy.deepcopy(frame_list[source_idx])
                x1, y1, x2, y2 = bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, ori_frame.shape[1])
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, ori_frame.shape[0])
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                except Exception:
                    continue
                
                # Merge results with version-specific parameters
                if args.version == "v15":
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                else:
                    combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
                cv2.imwrite(f"{result_img_save_path}/{str(write_idx).zfill(8)}.png", combine_frame)
                write_idx += 1

            if write_idx == 0:
                raise RuntimeError("Inference did not generate any output frames.")

            # Save prediction results
            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-v", "warning", "-r", str(fps),
                "-f", "image2", "-i", f"{result_img_save_path}/%08d.png",
                "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18",
                temp_vid_path
            ], check=True)

            subprocess.run([
                "ffmpeg", "-y", "-v", "warning",
                "-i", temp_vid_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                output_vid_name
            ], check=True)
            
            # Clean up temporary files
            shutil.rmtree(result_img_save_path)
            os.remove(temp_vid_path)
            
            if save_dir_full is not None and os.path.isdir(save_dir_full):
                shutil.rmtree(save_dir_full)
            if not (args.saved_coord or args.use_saved_coord) and os.path.exists(crop_coord_save_path):
                os.remove(crop_coord_save_path)
                    
            print(f"Results saved to {output_vid_name}")
        except Exception as e:
            print("Error occurred during processing:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")
    args = parser.parse_args()
    main(args)
