import argparse
from datetime import datetime
import json
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from torch.nn.functional import conv2d
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from typing import Any, Dict, List, Optional, Union
import os
import numpy as np
from tqdm import tqdm
import gc
import types
import shutil
import decord
import os
import random

from diffusers.models.attention import Attention
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.utils import WEIGHTS_NAME, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from transformers import AutoTokenizer, UMT5EncoderModel, T5EncoderModel, T5Tokenizer
from diffusers.models.transformers.transformer_wan import (
    WanTransformerBlock,
    WanTransformer3DModel,
)

decord.bridge.set_bridge("torch")


def seed_everything(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    ext_freqs=None,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    if ext_freqs is None:
        freqs = (
            1.0
            / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
            / linear_factor
        )  # [D/2]
    else:
        freqs = ext_freqs
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


class CustomWanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        u,
        v,
        ext_freqs_h=None,
        ext_freqs_w=None,
        theta: float = 10000.0,
        enable_smoothing: bool = True,
        enable_temporal_smoothing: bool = False,
        divisor: int = 16,
        mu: int = 21,
        std: int = 11,
        mu_time: int = 11,
        std_time: int = 5,
        enable_lookup_norm: bool = False,
        lookup_norm_thr: float = 1.0,
        num_frames: int = 13,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        H = 30
        W = 52
        # num_frames = 13

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        def _gaussian_blur(x, k=5, sigma=3.0):
            kernel_1d = torch.tensor(
                [math.exp(-((i - (k // 2)) ** 2) / (2 * sigma**2)) for i in range(k)], device=x.device, dtype=x.dtype
            )
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d[None, None, :, :].repeat(1, 1, 1, 1)
            return conv2d(x.unsqueeze(1), kernel_2d, padding=k // 2).squeeze(1)

        if enable_smoothing:
            # bilinear resize to 30x52 and normalize flow values
            u_ds = F.interpolate(u.unsqueeze(1), size=(30, 52), mode="bilinear", align_corners=False).squeeze(1)
            v_ds = F.interpolate(v.unsqueeze(1), size=(30, 52), mode="bilinear", align_corners=False).squeeze(1)

            # Spatial smoothing
            u_ds = _gaussian_blur(u_ds, k=mu, sigma=std)
            v_ds = _gaussian_blur(v_ds, k=mu, sigma=std)

            if enable_temporal_smoothing:
                kernel_size = mu_time
                sigma_t = std_time
                kernel_t = torch.tensor(
                    [math.exp(-((i - (kernel_size // 2)) ** 2) / (2 * sigma_t**2)) for i in range(kernel_size)],
                    device=u_ds.device,
                    dtype=u_ds.dtype,
                )
                kernel_t = kernel_t / kernel_t.sum()

                # Reshape to [1, 1, T, H, W] for temporal convolution
                u_t = u_ds.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
                v_t = v_ds.unsqueeze(0).unsqueeze(0)

                # Apply temporal smoothing
                u_smooth = F.conv3d(u_t, kernel_t.view(1, 1, kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0))
                v_smooth = F.conv3d(v_t, kernel_t.view(1, 1, kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0))

                # Reshape back to [T, H, W]
                u_lat = u_smooth.squeeze(0).squeeze(0) / divisor  # [T, H, W]
                v_lat = v_smooth.squeeze(0).squeeze(0) / divisor
            else:
                u_lat = u_ds / divisor
                v_lat = v_ds / divisor
        else:
            u_lat = (
                F.interpolate(u.unsqueeze(1), size=(30, 52), mode="bilinear", align_corners=False).squeeze(1) / divisor
            )
            v_lat = (
                F.interpolate(v.unsqueeze(1), size=(30, 52), mode="bilinear", align_corners=False).squeeze(1) / divisor
            )

        # Accumulate flow with bounds checking
        cum_u, cum_v = [], []
        u_c = torch.zeros_like(u_lat[0])  # [30,45]
        v_c = torch.zeros_like(v_lat[0])

        if enable_lookup_norm:
            threshold = lookup_norm_thr
            eps = 1e-6
            cum_u, cum_v = [], []
            u_c = torch.zeros_like(u_lat[0])
            v_c = torch.zeros_like(v_lat[0])

            for t in range(u_lat.shape[0]):
                if t < u_lat.shape[0] - 1:
                    du = (u_lat[t + 1] - u_lat[t]).norm()
                    dv = (v_lat[t + 1] - v_lat[t]).norm()
                    diff = torch.sqrt(du * du + dv * dv)
                else:
                    diff = torch.tensor(0.0, dtype=u_lat.dtype, device=u_lat.device)

                w = torch.clamp(threshold / (diff + eps), max=1.0)

                u_c = u_c + w * u_lat[t]
                v_c = v_c + w * v_lat[t]
                cum_u.append(u_c.clone())
                cum_v.append(v_c.clone())

            cum_u = torch.stack(cum_u)
            cum_v = torch.stack(cum_v)
        else:
            for t in range(u_lat.shape[0]):  # F frames
                u_c = u_c + u_lat[t]
                v_c = v_c + v_lat[t]
                cum_u.append(u_c.clone())
                cum_v.append(v_c.clone())
            cum_u = torch.stack(cum_u)  # [F, 30, 45]
            cum_v = torch.stack(cum_v)

        # Create base coordinate grid
        grid_h_custom, grid_w_custom = torch.meshgrid(
            torch.arange(30, device=u.device), torch.arange(52, device=u.device), indexing="ij"
        )
        # Apply accumulated displacement with smoothing
        h_motion = grid_h_custom[None] + cum_v  # [T,H,W]
        w_motion = grid_w_custom[None] + cum_u  # [T,H,W]

        # TODO quantize h_motion and w_motion in no-training pipeline

        h_motion_reshaped = h_motion.permute(0, 2, 1).reshape(-1, H)  # [T*W, H]
        w_motion_reshaped = w_motion.permute(0, 1, 2).reshape(-1, W)  # [T*H, W]

        # default case
        self.orig_freq_t = get_1d_rotary_pos_embed(
            dim=t_dim,
            pos=max_seq_len,
            theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )
        self.orig_freq_h = get_1d_rotary_pos_embed(
            dim=h_dim,
            pos=max_seq_len,
            theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )
        self.orig_freq_w = get_1d_rotary_pos_embed(
            dim=w_dim,
            pos=max_seq_len,
            theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            freqs_dtype=torch.float64,
        )

        # Get rotary embeddings for each row/column
        h_list = []
        w_list = []

        for i in range(h_motion_reshaped.size(0)):  # For each T*W rows
            if ext_freqs_h is not None:
                ext_freqs = ext_freqs_h[i]
            else:
                ext_freqs = None
            h_i = get_1d_rotary_pos_embed(
                dim=h_dim,
                pos=h_motion_reshaped[i],
                theta=theta,
                ext_freqs=ext_freqs,
                use_real=False,
                repeat_interleave_real=False,
            )
            h_list.append(h_i)

        for i in range(w_motion_reshaped.size(0)):  # For each T*H columns
            if ext_freqs_w is not None:
                ext_freqs = ext_freqs_w[i]
            else:
                ext_freqs = None
            w_i = get_1d_rotary_pos_embed(
                dim=w_dim,
                pos=w_motion_reshaped[i],
                theta=theta,
                ext_freqs=ext_freqs,
                use_real=False,
                repeat_interleave_real=False,
            )
            w_list.append(w_i)

        # Stack and reshape back to [T,H,W,dim]
        self.h_freqs = torch.stack(h_list).reshape(num_frames, W, H, -1).permute(0, 2, 1, 3)  # [T,H,W,dim_h]
        self.w_freqs = torch.stack(w_list).reshape(num_frames, H, W, -1)  # [T,H,W,dim_w]
        self.t_freqs = get_1d_rotary_pos_embed(
            t_dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        freqs_f = self.t_freqs[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = self.h_freqs
        freqs_w = self.w_freqs

        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


def extract_rgb_flow(of_model, video_frames, flow2img=True):
    flow_feats = []
    for i in range(len(video_frames) - 1):
        flow = of_model(video_frames[i : i + 1], video_frames[i + 1 : i + 2])[-1]
        flow_feats.append(flow)
    flow_feats = torch.cat(flow_feats, dim=0)
    if flow2img:
        flow_feats = flow_to_image(flow_feats).to(video_frames.dtype) / 127.5 - 1  # [-1,1]
    return flow_feats


def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: Optional[int],
) -> torch.FloatTensor:
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        start_frame = min(skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - skip_frames_end)

        if end_frame <= start_frame:
            indices = [start_frame]
        elif end_frame - start_frame <= max_num_frames:
            indices = list(range(start_frame, end_frame))
        else:
            step = frame_sample_step or (end_frame - start_frame) // max_num_frames
            indices = list(range(start_frame, end_frame, step))

        frames = video_reader.get_batch(indices=indices)
        frames = frames[:max_num_frames].float()  # ensure that we don't go over the limit

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        selected_num_frames = frames.size(0)
        remainder = (3 + selected_num_frames) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        assert frames.size(0) % 4 == 1

        # Normalize the frames
        transform = T.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]


def encode_video(vae, video, device):
    # video: [F, C, H, W] → [1, C, F, H, W]
    if video.dim() == 4:
        video = video.unsqueeze(0)  # [1, F, C, H, W]
        video = video.permute(0, 2, 1, 3, 4)  # → [1, C, F, H, W]
    elif video.shape[0] == 3:  # [C, F, H, W]
        video = video.unsqueeze(0)  # [1, C, F, H, W]
    elif video.dim() == 5:
        video = video.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}")

    video = video.to(device, dtype=vae.dtype)

    latent_dist = vae.encode(video).latent_dist
    latents = latent_dist.sample()

    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = (latents - latents_mean) * latents_std  # match decode side
    return latents.to(memory_format=torch.contiguous_format).float()  # shape: [1, z_dim, F, H, W]


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


class CustomWanAttnProcessor2_0(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        custom_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if custom_rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, custom_rotary_emb)
            key = apply_rotary_emb(key, custom_rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomWanTransformerBlock(WanTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        custom_rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention with custom rotary embs
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states, rotary_emb=rotary_emb, custom_rotary_emb=custom_rotary_emb
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


def custom_transformer_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    rotary_emb: Optional[torch.Tensor] = None,
    custom_rotary_emb: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            print("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block_idx, block in enumerate(self.blocks):
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, custom_rotary_emb
            )
    else:
        for block_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, custom_rotary_emb)

    # 5. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


class WanModuleUtils:
    @staticmethod
    def modify_transformer_forward(model):
        model.forward = types.MethodType(custom_transformer_forward, model)
        return model

    @staticmethod
    def modify_transformer_layers(model):
        config = model.config

        num_attention_heads = config.num_attention_heads
        attention_head_dim = config.attention_head_dim
        ffn_dim = config.ffn_dim
        qk_norm = config.qk_norm
        cross_attn_norm = config.cross_attn_norm
        eps = config.eps
        added_kv_proj_dim = config.added_kv_proj_dim
        num_layers = config.num_layers
        inner_dim = num_attention_heads * attention_head_dim

        ## Set custom transformer blocks
        for idx in range(num_layers):
            b = CustomWanTransformerBlock(
                inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
            )
            b.load_state_dict(model.blocks[idx].state_dict(), strict=True)  # restore original weights
            model.blocks[idx] = b.to(device=model.device, dtype=model.dtype)

        ## Set attention processors
        for out_i in range(num_layers):
            processor = CustomWanAttnProcessor2_0()
            model.blocks[out_i].attn1.set_processor(processor)

        return model


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: Union[UMT5EncoderModel, T5EncoderModel],
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: Union[UMT5EncoderModel, T5EncoderModel],
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def phase_constraint_loss(
    pred, tgt, eps: float = 1e-8, norm: str = "ortho"
) -> torch.Tensor:

    # 1. 3-D FFT (t, h, w).  rfftn saves half the memory if inputs are real.
    Fp = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm=norm)
    Ft = torch.fft.rfftn(tgt, dim=(-3, -2, -1), norm=norm)

    # 2. Convert to unit-magnitude complex numbers  z = e^{jθ}
    zp = Fp / (Fp.abs() + eps)  # (B,T,C,H,W//2+1)
    zt = Ft / (Ft.abs() + eps)

    # 4. Smooth phase loss = L1 on real & imag parts
    loss = F.l1_loss(zp.real, zt.real) + F.l1_loss(zp.imag, zt.imag)
    return loss


def tune_p(
    ## Models
    vae,
    transformer,
    tokenizer,
    text_encoder,
    scheduler,
    ## Inputs
    video_path: str,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    ## Constants
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    ## Generation
    latents: Optional[torch.Tensor] = None,
    num_frames: int = 81,
    mse_weight=1.0,
    guidance_scale: float = 5.0,
    ## RoPE Warping Args
    train_mode: bool = True,
    tuned_rope_dtype: Optional[torch.dtype] = torch.complex64,
    divisor: int = 16,
    enable_smoothing: bool = True,
    mu: int = 21,
    std: int = 11,
    mu_time: int = 21,
    std_time: int = 11,
    enable_lookup_norm: bool = False,
    lookup_norm_thr: float = 1.0,
    n_replace_gt_mod: list = [],
    num_optim_steps: int = 5,
    tune_wo_warped_uv: bool = False,
    theta: float = 10000.0,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    ## Others
    output_type: Optional[str] = "np",
    return_dict: bool = True,
):
    weights = Raft_Small_Weights.DEFAULT
    of_model = raft_small(weights=weights).to("cuda")
    of_model.eval()

    skip_frames_start = 0
    skip_frames_end = 0
    max_num_frames = num_frames
    frame_sample_step = None

    prompt_embeds = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        prompt,
        226,
        "cuda",
        torch.float32,
        requires_grad=False,
    ).to("cuda")
    negative_prompt_embeds = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        negative_prompt,
        226,
        "cuda",
        torch.float32,
        requires_grad=False,
    ).to("cuda")

    transformer_dtype = transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    scheduler.set_timesteps(num_inference_steps, device="cuda")
    timesteps = scheduler.timesteps

    with torch.no_grad():
        print("Reading reference video")
        video_frames = get_video_frames(
            video_path=video_path,
            width=width,
            height=height,
            skip_frames_start=skip_frames_start,
            skip_frames_end=skip_frames_end,
            max_num_frames=max_num_frames,
            frame_sample_step=frame_sample_step,
        ).to(device="cuda")
        vid_indices = torch.linspace(0, video_frames.size(0) - 1, num_frames).long()
        video_frames = video_frames[vid_indices]
        print("Reference video frames shape:", video_frames.shape)

        latent_video = encode_video(vae=vae, video=video_frames, device="cuda")
        print("Latent video frames shape:", latent_video.shape)

        gt_flow_feats = extract_rgb_flow(of_model=of_model, video_frames=video_frames, flow2img=False)
        gt_flow_feats = gt_flow_feats.to(device="cuda", dtype=torch.float32)

        indices = torch.linspace(0, gt_flow_feats.size(0) - 1, 13 if num_frames == 49 else 21).long()
        gt_flow_feats = gt_flow_feats[indices]

        custom_rope = CustomWanRotaryPosEmbed(
            attention_head_dim=transformer.config.attention_head_dim,
            patch_size=transformer.config.patch_size,
            max_seq_len=transformer.config.rope_max_seq_len,
            u=gt_flow_feats[:, 0, :, :].to("cpu"),
            v=gt_flow_feats[:, 1, :, :].to("cpu"),
            divisor=divisor,
            enable_smoothing=enable_smoothing,
            theta=theta,
            mu=mu,
            std=std,
            mu_time=mu_time,
            std_time=std_time,
            enable_lookup_norm=enable_lookup_norm,
            lookup_norm_thr=lookup_norm_thr,
            # num_frames=13 if num_frames == 49 else 21,
        )
        custom_image_rotary_emb = custom_rope(latents).to(device="cuda")
        gt_image_rotary_emb = transformer.rope(latents).to(device="cuda")

        if tune_wo_warped_uv:
            custom_image_rotary_emb = gt_image_rotary_emb

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    of_model.requires_grad_(False)
    del text_encoder
    del of_model
    clean_memory()

    video_processor = VideoProcessor(vae_scale_factor=8)

    if n_replace_gt_mod:
        if train_mode:
            tunable_image_rotary_emb = [
                torch.nn.Parameter(torch.zeros_like(custom_image_rotary_emb).to(dtype=tuned_rope_dtype))
                for _ in range(len(n_replace_gt_mod))
            ]
            optimizer = torch.optim.AdamW(tunable_image_rotary_emb, lr=1e-3)
        else:
            tunable_image_rotary_emb = [torch.zeros_like(custom_image_rotary_emb) for _ in range(len(n_replace_gt_mod))]

    for i, t in tqdm(enumerate(timesteps), desc="Inference Timestep", total=len(timesteps), leave=True, position=0):
        latent_model_input = latents.to(transformer_dtype)
        timestep = t.expand(latents.shape[0]).to("cuda")

        if i in n_replace_gt_mod:
            if train_mode:
                sigmas = scheduler.sigmas.to("cuda")[i]
                while sigmas.ndim < latent_model_input.ndim:
                    sigmas = sigmas.unsqueeze(-1)

                progress_bar = tqdm(
                    range(num_optim_steps), desc="Optimizing Step", leave=False, position=1, total=num_optim_steps
                )
                for optim_iter in range(num_optim_steps):
                    delta_p = sum(tunable_image_rotary_emb[: i + 1])

                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        custom_rotary_emb=custom_image_rotary_emb + delta_p,
                        # custom_rotary_emb=custom_image_rotary_emb * torch.prod(torch.stack(tunable_image_rotary_emb[:i+1]), dim=0),
                    )[0]

                    target_velocity = (latent_model_input - latent_video) / sigmas

                    loss = phase_constraint_loss(
                        noise_pred, target_velocity
                    ) + mse_weight * torch.nn.functional.mse_loss(noise_pred, target_velocity, reduction="mean")

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    clean_memory()

                    progress_bar.set_description(f"Step {optim_iter + 1}: loss {loss.item():.6f}")

                    progress_bar.update(1)

                progress_bar.close()

            # train mode or not, perform forward passes to get x_t-1
            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=custom_image_rotary_emb + delta_p,
                )[0]

                noise_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=custom_image_rotary_emb + delta_p,
                )[0]

                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        else:
            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=gt_image_rotary_emb,
                )[0]

                noise_uncond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    custom_rotary_emb=gt_image_rotary_emb,
                )[0]

                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    if not output_type == "latent":
        latents = latents.to(vae.dtype)
        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = vae.decode(latents, return_dict=False)[0]
        video = video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    if not return_dict:
        return (video,)

    return WanPipelineOutput(frames=video)

# Global variables and model loading
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 832
NUM_INFERENCE_STEPS = 50

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32,).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = UMT5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to("cuda")
transformer = WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.float32).to("cuda")
 
transformer = WanModuleUtils.modify_transformer_layers(transformer)
transformer = WanModuleUtils.modify_transformer_forward(transformer)
transformer.enable_gradient_checkpointing()

seed_everything(42)
latents = torch.randn(1, 16, 13, 60, 104).to("cuda")

def generate(args):

    output_dir = args.output_dir
    video_path = args.input_video
    prompt = args.prompt

    assert os.path.isfile(video_path), f"Input video file {video_path} does not exist."

    output_dir += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(__file__, os.path.join(output_dir, os.path.basename(__file__)))

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    output = tune_p(
        ## Models
        vae=vae,
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        ## Inputs
        video_path=video_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        ## Constants
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        ## Generation
        latents=latents.clone(),
        num_frames=args.frames,        
        mse_weight=args.mse_weight,
        guidance_scale=5.0,
        ## RoPE Warping Args
        train_mode=True,
        tuned_rope_dtype=torch.complex64,
        divisor=8,  # 16
        enable_smoothing=True,  # True
        mu=21,
        std=11,  ## 21,11
        mu_time=5,
        std_time=3,  # 11,5
        enable_lookup_norm=True,
        lookup_norm_thr=1.0,
        n_replace_gt_mod=range(args.n_replace_gt_mod),
        num_optim_steps=args.num_optim_steps,
        tune_wo_warped_uv=(1 - args.start_with_uv_warped),
    ).frames[0]

    export_to_video(output, f"{output_dir}/{prompt_clean(prompt).replace(' ', '_')[:20]}.mp4", fps=15)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video",
        type=str,
        default="./input_video.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat playing with a ball of yarn",
        help="Text prompt to guide the video generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ropecraft_results",
        help="Directory to dump the results",
    )
    parser.add_argument(
        "--start_with_uv_warped",
        type=int,
        default=1,
        choices=[0, 1],
        help="Start with UV warped delta_p (0=False, 1=True)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=49,
        choices=[49, 81],
        help="Number of frames to process",
    )
    parser.add_argument(
        "--num_optim_steps",
        type=int,
        default=5,
        help="Number of optimization steps for tuning",
    )
    parser.add_argument(
        "--n_replace_gt_mod",
        type=int,
        default=10,
        help="Number of optimization steps for tuning",
    )
    parser.add_argument(
        "--mse_weight",
        type=float,
        default=0.1,
        help="Weight for MSE loss",
    )

    args = parser.parse_args()

    print("Using arguments:", args)

    generate(args)
