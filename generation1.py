import torch
import os
import cv2
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder, StableVideoDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image, ImageOps
import ffmpeg

def resize_and_pad_image(image_path, target_size=(1024, 576)):
    image = Image.open(image_path).convert("RGB")
    
    image.thumbnail(target_size, Image.LANCZOS)
    
    delta_w = target_size[0] - image.size[0]
    delta_h = target_size[1] - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    image = ImageOps.expand(image, padding, (0, 0, 0))
    return image

image_dir = '/scratch/shengjie/svdtrain/YU/test'
output_generated_dir = '/scratch/shengjie/svdtrain/YU/evalG'
os.makedirs(output_generated_dir, exist_ok=True)

unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "/scratch/shengjie/svdtrain/fine-tuned-model-final/unet",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    "/scratch/shengjie/svdtrain/fine-tuned-model-final/vae",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
vision_model = CLIPVisionModelWithProjection.from_pretrained(
    "/scratch/shengjie/svdtrain/fine-tuned-model-final/vision_model",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)

feature_extractor = CLIPImageProcessor.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    subfolder="feature_extractor",
    local_files_only=False,
)
scheduler = EulerDiscreteScheduler.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    subfolder="scheduler",
    local_files_only=False,
)

pipe = StableVideoDiffusionPipeline(
    unet=unet,
    vae=vae,
    image_encoder=vision_model,
    feature_extractor=feature_extractor,
    scheduler=scheduler
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe.to(device)

group_name = "images1"
image_filename = f"{group_name}_frame_1.png"
image_path = os.path.join(image_dir, image_filename)

image = resize_and_pad_image(image_path)

generator = torch.manual_seed(-1)
total_frames = 60 

all_frames = []
with torch.inference_mode():
    image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    image_embeddings = vision_model(image_tensor).image_embeds

    frames = pipe(
        image,
        num_frames=total_frames,
        width=1024,
        height=576,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=127,
        fps=6,
        num_inference_steps=100,
    ).frames[0]
    all_frames.extend(frames)

temp_frame_dir = os.path.join(output_generated_dir, "temp_frames")
os.makedirs(temp_frame_dir, exist_ok=True)

for i, frame in enumerate(all_frames):
    frame_path = os.path.join(temp_frame_dir, f"frame_{i:04d}.png")
    frame.save(frame_path)

video_path = os.path.join(output_generated_dir, f"{group_name}.mp4")
(
    ffmpeg
    .input(os.path.join(temp_frame_dir, 'frame_%04d.png'), framerate=6)
    .output(video_path, pix_fmt='yuv420p', video_bitrate='5M')
    .run()
)

for frame_file in os.listdir(temp_frame_dir):
    os.remove(os.path.join(temp_frame_dir, frame_file))
os.rmdir(temp_frame_dir)

print(f"Generated video saved to {video_path}")
