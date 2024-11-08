import os
import time
from pathlib import Path
import random
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
import PIL
from PIL import Image
import numpy as np
import cv2
import shutil
import logging
import math
from tqdm.auto import tqdm

import sys
sys.path.append('/scratch/shengjie/svdtrain/src')
from unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

logger = get_logger(__name__, log_level="INFO")

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

device = set_device()

class VideoFrameDataset(Dataset):
    def __init__(self, video_dir, frames_per_video=5, width=1024, height=576):
        self.video_dir = Path(video_dir)
        self.frames_per_video = frames_per_video
        self.width = width
        self.height = height
        self.channels = 3
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        video_ids = set([file.stem.rsplit('_frame', 1)[0] for file in self.video_dir.glob("*.png")])
        for video_id in video_ids:
            frames = self._load_frames(video_id)
            if frames:
                data.append({"frames": frames, "video_id": video_id})
        return data

    def _load_frames(self, video_id):
        frames = []
        for idx in range(self.frames_per_video):
            frame_file = self.video_dir / f"{video_id}_frame_{idx}.png"
            frame = Image.open(frame_file)
            frame = frame.convert("RGB")
            frame = frame.resize((self.width, self.height))
            frame = np.array(frame).astype(np.float32) / 127.5 - 1
            frames.append(frame)
        return frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = item["frames"]
        pixel_values = torch.empty((self.frames_per_video, self.channels, self.height, self.width))
        
        for i, frame in enumerate(frames):
            frame_tensor = torch.tensor(frame).permute(2, 0, 1)
            pixel_values[i] = frame_tensor
        
        return {"pixel_values": pixel_values}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0).to(device)
    return {"pixel_values": pixel_values}

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    
    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)

def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]
    padding_shape = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    out = output.view(b, c, h, w)
    return out

def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

def _compute_padding(kernel_size):
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)
    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
            f"but a vector of {passed_add_embed_dim} was created. Please check unet.config.time_embedding_type "
            f"and text_encoder_2.config.projection_dim."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype).to(device)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids

processor = CLIPImageProcessor.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="feature_extractor")
train_dataset = VideoFrameDataset('/scratch/shengjie/svdtrain/YU/train', frames_per_video=5)
val_dataset = VideoFrameDataset('/scratch/shengjie/svdtrain/YU/val', frames_per_video=5)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "/scratch/shengjie/svdtrain/models/unet",
    low_cpu_mem_usage=False,
)

vae = AutoencoderKLTemporalDecoder.from_pretrained(
    "/scratch/shengjie/svdtrain/models/vae",
    low_cpu_mem_usage=False,
)

vision_model = CLIPVisionModelWithProjection.from_pretrained(
    "/scratch/shengjie/svdtrain/models/image_encoder",
    low_cpu_mem_usage=False,
)

processor = CLIPImageProcessor.from_pretrained(
    "/scratch/shengjie/svdtrain/models/feature_extractor",
    local_files_only=True,
)

scheduler = EulerDiscreteScheduler.from_pretrained(
    "/scratch/shengjie/svdtrain/models/scheduler",
    local_files_only=True,
)

vae.requires_grad_(False)
vision_model.requires_grad_(False)

for name, param in unet.named_parameters():
    if 'temporal_transformer_block' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = AdamW(unet.parameters(), lr=1e-5)
scaler = GradScaler()

ema_unet = EMAModel(unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

accelerator = Accelerator(mixed_precision='no')

train_dataloader, val_dataloader, unet, optimizer, scaler, vision_model, vae = accelerator.prepare(
    train_dataloader, val_dataloader, unet, optimizer, scaler, vision_model, vae
)

num_epochs = 5
accumulation_steps = 32

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w").to(device)
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents

def encode_image(pixel_values):
    pixel_values = pixel_values[:, 0, :, :, :]
    pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
    pixel_values = (pixel_values + 1.0) / 2.0
    pixel_values = processor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt"
    ).pixel_values
    pixel_values = pixel_values.to(device)
    image_embeddings = vision_model(pixel_values=pixel_values).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1).repeat(1, 5, 1)
    return image_embeddings

def evaluate_model(dataloader, model, vision_model, vae, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs in dataloader:
            pixel_values = inputs["pixel_values"].to(device)
            image_embeddings = encode_image(pixel_values.float()).to(device)

            latents = tensor_to_vae_latent(pixel_values, vae).to(device)

            bsz = latents.shape[0]
            cond_sigmas = rand_log_normal(shape=[bsz], loc=-3.0, scale=0.5).to(latents.device)
            noise_aug_strength = cond_sigmas[0]
            cond_sigmas = cond_sigmas[:, None, None, None, None]
            conditional_pixel_values = torch.randn_like(pixel_values).to(device) * cond_sigmas + pixel_values
            conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
            conditional_latents = conditional_latents / vae.config.scaling_factor

            noise = torch.randn_like(latents).to(device)
            sigmas = rand_log_normal(shape=[latents.shape[0]], loc=0.7, scale=1.6, device=device).to(device)
            sigmas = sigmas[:, None, None, None, None]
            noisy_latents = latents + noise * sigmas
            timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(device)

            added_time_ids = _get_add_time_ids(fps=7, motion_bucket_id=127, noise_aug_strength=0.1,
                                               dtype=latents.dtype, batch_size=latents.shape[0])

            inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
            conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
            inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

            noise_pred = model(inp_noisy_latents, timesteps, encoder_hidden_states=image_embeddings, 
                               added_time_ids=added_time_ids).sample

            c_out = -sigmas / ((sigmas**2 + 1)**0.5)
            c_skip = 1 / (sigmas**2 + 1)
            denoised_latents = noise_pred * c_out + c_skip * noisy_latents

            loss = F.mse_loss(denoised_latents.float(), latents.float())
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss

for epoch in range(num_epochs):
    total_loss = 0.0
    epoch_start_time = time.time()
    optimizer.zero_grad()

    for i, inputs in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            image_embeddings = encode_image(pixel_values.float()).to(device)

        latents = tensor_to_vae_latent(pixel_values, vae).to(device)
        bsz = latents.shape[0]
        cond_sigmas = rand_log_normal(shape=[bsz], loc=-3.0, scale=0.5).to(latents.device)
        noise_aug_strength = cond_sigmas[0]
        cond_sigmas = cond_sigmas[:, None, None, None, None]

        conditional_pixel_values = torch.randn_like(pixel_values).to(device) * cond_sigmas + pixel_values
        conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
        conditional_latents = conditional_latents / vae.config.scaling_factor

        noise = torch.randn_like(latents).to(device)
        sigmas = rand_log_normal(shape=[latents.shape[0]], loc=0.7, scale=1.6).to(latents.device)
        sigmas = sigmas[:, None, None, None, None]
        noisy_latents = latents + noise * sigmas
        timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(device)

        added_time_ids = _get_add_time_ids(fps=7, motion_bucket_id=127, noise_aug_strength=noise_aug_strength,
                                           dtype=latents.dtype, batch_size=latents.shape[0])

        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
        conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
        inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

        noise_pred = unet(inp_noisy_latents, timesteps, encoder_hidden_states=image_embeddings, 
                          added_time_ids=added_time_ids).sample

        c_out = -sigmas / ((sigmas**2 + 1)**0.5)
        c_skip = 1 / (sigmas**2 + 1)
        denoised_latents = noise_pred * c_out + c_skip * noisy_latents

        loss = F.mse_loss(denoised_latents.float(), latents.float())

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}: {100 * (epoch + 1) / num_epochs:.2f}% | Avg. Train Loss: {avg_train_loss:.4f} | Time: {time.time() - epoch_start_time:.2f}s")
    avg_val_loss = evaluate_model(val_dataloader, unet, vision_model, vae, device)
    print(f"Epoch {epoch + 1}/{num_epochs}: {100 * (epoch + 1) / num_epochs:.2f}% | Avg. Val Loss: {avg_val_loss:.4f} | Time: {time.time() - epoch_start_time:.2f}s")

    if (epoch + 1) % 10 == 0:
        accelerator.wait_for_everyone()
        unet = accelerator.unwrap_model(unet)
        vae = accelerator.unwrap_model(vae)
        vision_model = accelerator.unwrap_model(vision_model)
        
        save_dir = f"fine-tuned-model-epoch-{epoch + 1}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        unet.save_pretrained(os.path.join(save_dir, "unet"))
        vae.save_pretrained(os.path.join(save_dir, "vae"))
        vision_model.save_pretrained(os.path.join(save_dir, "vision_model"))

accelerator.wait_for_everyone()
unet = accelerator.unwrap_model(unet)
vae = accelerator.unwrap_model(vae)
vision_model = accelerator.unwrap_model(vision_model)

final_save_dir = 'fine-tuned-model-final'
if not os.path.exists(final_save_dir):
    os.makedirs(final_save_dir)

unet.save_pretrained(os.path.join(final_save_dir, "unet"))
vae.save_pretrained(os.path.join(final_save_dir, "vae"))
vision_model.save_pretrained(os.path.join(final_save_dir, "vision_model"))

print("Training completed.")
