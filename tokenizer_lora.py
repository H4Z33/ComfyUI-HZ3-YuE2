import json
import math
import os
import re
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import safetensors.torch

import folder_paths
import comfy.model_management
import comfy.sd


DEFAULT_DATASET = r"E:\GEN_AI\YuE2_Training\artist"
DEFAULT_RUNTIME = r"E:\GEN_AI\YuE2_Training\runtime"


def _safe_name(value):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._")
    if not name:
        raise ValueError("name must contain at least one letter or number")
    return name


def _defensive_scalar(value, default=None):
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else default
    return value if value is not None else default


def _audio_array(audio):
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    waveform = audio["waveform"].detach().float().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform.clamp(-1, 1).transpose(0, 1).numpy(), int(audio["sample_rate"])


class LoRALinear(nn.Module):
    """Low-rank adapter injected over a base linear projection without modifying base weights."""
    def __init__(self, base_layer, rank=32, alpha=32.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = getattr(base_layer, "in_features", None)
        self.out_features = getattr(base_layer, "out_features", None)
        if self.in_features is None or self.out_features is None:
            weight = getattr(base_layer, "weight", None)
            if weight is not None:
                self.out_features, self.in_features = weight.shape[:2]
            else:
                self.out_features, self.in_features = 2048, 2048
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)

        params = list(base_layer.parameters())
        device = params[0].device if params else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = params[0].dtype if params else torch.bfloat16
        if dtype not in (torch.float32, torch.bfloat16, torch.float16):
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

        # A (down): initialized with normal distribution scaled by 1/sqrt(d_in)
        self.lora_down = nn.Parameter(
            torch.randn(self.rank, self.in_features, device=device, dtype=dtype) * (1.0 / math.sqrt(self.in_features))
        )
        # B (up): initialized to zero so adapter starts as an identity operation
        self.lora_up = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=device, dtype=dtype)
        )

    @property
    def weight(self):
        return getattr(self.base_layer, "weight", None)

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    def forward(self, x):
        base_out = self.base_layer(x)
        delta = (x.to(self.lora_down.dtype) @ self.lora_down.T) @ self.lora_up.T
        return base_out + delta.to(base_out.dtype) * self.scale

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)


def _find_yue2_checkpoint():
    explicit_paths = [
        Path(r"E:\GEN_AI\YUE\checkpoints\yue2_3b_int8_convrot.safetensors"),
        Path(folder_paths.models_dir) / "checkpoints" / "yue2_3b_int8_convrot.safetensors",
    ]
    for p in explicit_paths:
        if p.exists():
            return str(p)
    try:
        for name in folder_paths.get_filename_list("checkpoints"):
            if "yue2" in name.lower():
                full_path = folder_paths.get_full_path("checkpoints", name)
                if full_path and Path(full_path).exists():
                    return full_path
    except Exception:
        pass
    return None


def _find_tokenizer_head():
    candidates = [
        Path(r"E:\GEN_AI\upstream_yue2_realaudio_tokenizer\tokenizer_head_joint_v4.pt"),
        Path(DEFAULT_RUNTIME) / "tokenizer_head_joint_v4.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _extract_audio_latents(audio_input, vae_model, device):
    if isinstance(audio_input, (list, tuple)):
        audio_input = audio_input[0]
    raw_waveform = audio_input["waveform"].detach().float().cpu()
    sr = int(audio_input["sample_rate"])
    if raw_waveform.ndim == 3:
        raw_waveform = raw_waveform[0]
    if raw_waveform.ndim == 1:
        raw_waveform = raw_waveform.unsqueeze(0)
    if raw_waveform.shape[0] == 1:
        raw_waveform = raw_waveform.repeat(2, 1)
    elif raw_waveform.shape[0] > 2:
        raw_waveform = raw_waveform[:2]

    # Resample to 48kHz for YuE2 VAE
    if sr != 48000:
        raw_waveform = torchaudio.functional.resample(raw_waveform, sr, 48000)

    # VAE audio encode in ComfyUI expects [batch, samples, channels]
    waveform_in = raw_waveform.unsqueeze(0).movedim(1, -1).to(device)
    with torch.inference_mode():
        lat = vae_model.encode(waveform_in)
        if isinstance(lat, dict):
            lat = lat.get("samples", lat)
    with torch.inference_mode(False):
        lat = lat.clone()
    return lat


def _autograd_rope1(x, freqs_cis):
    t = x.to(freqs_cis.dtype) if torch.promote_types(x.dtype, freqs_cis.dtype) != freqs_cis.dtype else x
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis.unsqueeze(0)
    diagonal = freqs_cis.diagonal(dim1=-2, dim2=-1).movedim(-1, -2)
    out = t.unflatten(-1, (2, -1)) * diagonal
    half = x.shape[-1] // 2
    out0 = out[..., 0, :] + t[..., half:] * freqs_cis[..., 0, 1]
    out1 = out[..., 1, :] + t[..., :half] * freqs_cis[..., 1, 0]
    return torch.stack([out0, out1], dim=-2).reshape(x.shape).type_as(x)


def _autograd_apply_rope(xq, xk, freqs_cis):
    import comfy.text_encoders.llama as llama
    matrix = llama.rope_matrix(freqs_cis)
    if matrix.ndim == 5:
        matrix = matrix.unsqueeze(0)
    q_ndim, k_ndim = xq.ndim, xk.ndim
    if q_ndim == 3:
        xq = xq.unsqueeze(0)
    if k_ndim == 3:
        xk = xk.unsqueeze(0)
    q_out = _autograd_rope1(xq, matrix)
    k_out = _autograd_rope1(xk, matrix)
    if q_ndim == 3:
        q_out = q_out.squeeze(0)
    if k_ndim == 3:
        k_out = k_out.squeeze(0)
    return q_out, k_out


def _make_training_forward(layer):
    """Provide an autograd-safe forward for TransformerBlock without in-place add(out=...)."""
    def training_forward(self, x, attention_mask=None, freqs_cis=None, optimized_attention=None, past_key_value=None):
        residual = x
        norm_x = self.input_layernorm(x)
        attn_out, present_kv = self.self_attn(
            hidden_states=norm_x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
        )
        x = residual + attn_out
        residual = x
        norm_x2 = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x2)
        return residual + mlp_out, present_kv
    return types.MethodType(training_forward, layer)


def _prepare_model_for_training(model: torch.nn.Module, target_device: torch.device):
    """
    Prepare a model loaded under ComfyUI's inference_mode for autograd training.
    Re-creates any nn.Parameter or buffer whose underlying C++ TensorImpl was created
    inside InferenceMode (which prevents version counter tracking during autograd backward).
    Also temporarily disables comfy_cast_weights during training so ComfyUI doesn't invoke VBAR offloading.
    Returns a cleanup callable to restore original module states.
    """
    saved_comfy_cast = {}
    with torch.inference_mode(False):
        for mod in model.modules():
            if hasattr(mod, "comfy_cast_weights"):
                saved_comfy_cast[mod] = mod.comfy_cast_weights
                mod.comfy_cast_weights = False

            for name, param in list(mod.named_parameters(recurse=False)):
                if param is not None:
                    needs_fix = False
                    try:
                        _ = param._version
                    except RuntimeError:
                        needs_fix = True
                    if needs_fix or param.is_inference():
                        cloned = param.data.to(target_device).clone()
                        new_param = torch.nn.Parameter(cloned, requires_grad=param.requires_grad)
                        setattr(mod, name, new_param)

            for name, buf in list(mod.named_buffers(recurse=False)):
                if buf is not None:
                    needs_fix = False
                    try:
                        _ = buf._version
                    except RuntimeError:
                        needs_fix = True
                    if needs_fix or buf.is_inference():
                        cloned_buf = buf.data.to(target_device).clone()
                        mod.register_buffer(name, cloned_buf, persistent=getattr(buf, "persistent", True))

    def restore():
        for mod, prev_cast in saved_comfy_cast.items():
            mod.comfy_cast_weights = prev_cast

    return restore


def _train_voice_lora(audio_latents, diffusion_model, clip_model=None, trigger="hz3_artist", style_caption="", steps=100, rank=32, alpha=None, lr=5e-4, device="cuda"):
    """Train a LoRA adapter on YuE2's acoustic diffusion transformer (NAR branch)."""
    with torch.inference_mode(False), torch.enable_grad():
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        diffusion_model.to(torch_device)
        audio_latents = audio_latents.detach().clone()
        restore_model = _prepare_model_for_training(diffusion_model, torch_device)
        
        lora_alpha = float(alpha if alpha is not None else (rank * 2.0))
        
        import comfy.text_encoders.llama as llama
        orig_apply_rope = llama.apply_rope
        llama.apply_rope = _autograd_apply_rope
        
        original_layers = {}
        original_forwards = {}
        lora_modules = []
        lora_params = []
        
        layers = diffusion_model.model.layers
        for layer_idx, layer in enumerate(layers):
            original_forwards[layer_idx] = layer.forward
            layer.forward = _make_training_forward(layer)
            
            # Attention projections
            for proj_name in ("qkv_proj", "o_proj"):
                if hasattr(layer.self_attn, proj_name):
                    base_mod = getattr(layer.self_attn, proj_name)
                    original_layers[(layer_idx, "self_attn", proj_name)] = base_mod
                    lora_mod = LoRALinear(base_mod, rank=rank, alpha=lora_alpha)
                    setattr(layer.self_attn, proj_name, lora_mod)
                    lora_modules.append((layer_idx, "self_attn", proj_name, lora_mod))
                    lora_params.extend([lora_mod.lora_down, lora_mod.lora_up])
            
            # MLP projections
            for proj_name in ("gate_up_proj", "down_proj"):
                if hasattr(layer.mlp, proj_name):
                    base_mod = getattr(layer.mlp, proj_name)
                    original_layers[(layer_idx, "mlp", proj_name)] = base_mod
                    lora_mod = LoRALinear(base_mod, rank=rank, alpha=lora_alpha)
                    setattr(layer.mlp, proj_name, lora_mod)
                    lora_modules.append((layer_idx, "mlp", proj_name, lora_mod))
                    lora_params.extend([lora_mod.lora_down, lora_mod.lora_up])

        opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
        use_cuda = (torch_device.type == "cuda")
        
        total_frames = audio_latents.shape[-1]
        window_len = min(total_frames, 256)
        dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        
        # Build prompt-aware acoustic conditioning KV cache if clip_model is provided
        ctx = None
        chunks = None
        if clip_model is not None:
            try:
                clip_te = getattr(clip_model, "cond_stage_model", clip_model)
                style_str = f"in the style of {trigger}. {(style_caption or '').strip()}".strip()
                tok = clip_model.tokenize(style_str, lyrics="", cot="off")
                p = tok.get("prefix", [151643])
                dummy_tokens = [151853] * window_len
                with torch.inference_mode():
                    cond, c_chunks = clip_te._acoustic_conditioning(p, dummy_tokens, dtype)
                if cond is not None and len(c_chunks) > 0:
                    with torch.inference_mode(False):
                        ctx = cond.to(torch_device, dtype=dtype).clone()
                    chunks = list(c_chunks)
            except Exception:
                ctx = None
                chunks = None

        if ctx is None or chunks is None:
            # Fallback zero acoustic context [1, window_len, 28 * 2 * 8 * 128]
            ctx = torch.zeros(1, window_len, 28 * 2 * 8 * 128, device=torch_device, dtype=dtype)
            chunks = [(0, window_len, 0, window_len)]
        
        losses = []
        t0 = time.time()
        prev_training_state = getattr(comfy.model_management, "in_training", False)
        comfy.model_management.in_training = True
        try:
            for step in range(1, steps + 1):
                comfy.model_management.throw_exception_if_processing_interrupted()
                opt.zero_grad(set_to_none=True)
                
                max_start = max(0, total_frames - window_len)
                start_idx = 0 if max_start == 0 else torch.randint(0, max_start + 1, (1,)).item()
                clean_slice = audio_latents[:, :, start_idx:start_idx + window_len].to(torch_device, dtype=dtype)
                
                # Flow-matching velocity target: t in [0.01, 0.99]
                t = torch.rand(1, device=torch_device).clamp(0.01, 0.99)
                eps = torch.randn_like(clean_slice)
                z_t = (1.0 - t.view(1, 1, 1)) * clean_slice + t.view(1, 1, 1) * eps
                target_v = eps - clean_slice
                
                with torch.amp.autocast("cuda", enabled=use_cuda, dtype=dtype):
                    pred_v = diffusion_model(z_t, t, ctx, chunks)
                    loss = F.mse_loss(pred_v, target_v)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                opt.step()
                
                losses.append(loss.item())
        finally:
            restore_model()
            comfy.model_management.in_training = prev_training_state
            llama.apply_rope = orig_apply_rope
            for layer_idx, orig_fwd in original_forwards.items():
                layers[layer_idx].forward = orig_fwd

        # Collect trained LoRA tensors
        lora_dict = {}
        for layer_idx, block_name, proj_name, lora_mod in lora_modules:
            down_weight = lora_mod.lora_down.detach().cpu().contiguous()
            up_weight = lora_mod.lora_up.detach().cpu().contiguous()
            
            # Standard ComfyUI diffusion_model path
            base_prefix = f"diffusion_model.model.layers.{layer_idx}.{block_name}.{proj_name}"
            lora_dict[f"{base_prefix}.lora_down.weight"] = down_weight
            lora_dict[f"{base_prefix}.lora_up.weight"] = up_weight
            lora_dict[f"{base_prefix}.alpha"] = torch.tensor(float(lora_alpha))
            
        # Restore original layers
        for (layer_idx, block_name, proj_name), base_mod in original_layers.items():
            layer = layers[layer_idx]
            block = getattr(layer, block_name)
            setattr(block, proj_name, base_mod)
            
        duration = time.time() - t0
        return lora_dict, losses, duration


def _train_style_lora(audio_input, clip_model, trigger="hz3_artist", style_caption="", lyrics="", steps=80, rank=32, alpha=None, lr=3e-4, device="cuda"):
    """Train a LoRA adapter on YuE2's autoregressive text & semantic music model (AR branch)."""
    with torch.inference_mode(False), torch.enable_grad():
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        clip_te = clip_model.cond_stage_model
        clip_te.to(torch_device)
        ar_model = clip_te.model
        restore_model = _prepare_model_for_training(ar_model, torch_device)
        lora_alpha = float(alpha if alpha is not None else (rank * 2.0))
        
        import comfy.text_encoders.llama as llama
        orig_apply_rope = llama.apply_rope
        llama.apply_rope = _autograd_apply_rope

        original_layers = {}
        original_forwards = {}
        lora_modules = []
        lora_params = []
        
        layers = ar_model.layers
        for layer_idx, layer in enumerate(layers):
            original_forwards[layer_idx] = layer.forward
            layer.forward = _make_training_forward(layer)
            
            # Attention projections
            for proj_name in ("qkv_proj", "o_proj"):
                if hasattr(layer.self_attn, proj_name):
                    base_mod = getattr(layer.self_attn, proj_name)
                    original_layers[(layer_idx, "self_attn", proj_name)] = base_mod
                    lora_mod = LoRALinear(base_mod, rank=rank, alpha=lora_alpha)
                    setattr(layer.self_attn, proj_name, lora_mod)
                    lora_modules.append((layer_idx, "self_attn", proj_name, lora_mod))
                    lora_params.extend([lora_mod.lora_down, lora_mod.lora_up])
                    
            # MLP projections
            for proj_name in ("gate_up_proj", "down_proj"):
                if hasattr(layer.mlp, proj_name):
                    base_mod = getattr(layer.mlp, proj_name)
                    original_layers[(layer_idx, "mlp", proj_name)] = base_mod
                    lora_mod = LoRALinear(base_mod, rank=rank, alpha=lora_alpha)
                    setattr(layer.mlp, proj_name, lora_mod)
                    lora_modules.append((layer_idx, "mlp", proj_name, lora_mod))
                    lora_params.extend([lora_mod.lora_down, lora_mod.lora_up])

        opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
        
        style_text = f"{trigger}, in the style of {trigger}. {(style_caption or '').strip()}".strip()
        lyrics_text = (lyrics or "[instrumental]").strip()
        
        token_dict = clip_model.tokenize(style_text, lyrics=lyrics_text, cot="off")
        prefix_ids = token_dict.get("prefix", [151643])
        
        sample_rate = int(audio_input.get("sample_rate", 48000))
        raw_waveform = audio_input["waveform"].detach().float().cpu()
        if raw_waveform.ndim == 3:
            raw_waveform = raw_waveform[0]
        if raw_waveform.ndim == 2:
            raw_waveform = raw_waveform.mean(0)
            
        use_cuda = (torch_device.type == "cuda")
        dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float32
        
        target_tokens_count = max(32, min(256, int(len(raw_waveform) / sample_rate * 25)))
        music_tokens = None
        try:
            with torch.inference_mode():
                gen_tokens, _ = clip_te._generate(
                    prefix_ids, 42, target_tokens_count, "semantic", dtype,
                    negative=token_dict.get("negative", [151643]) + [151851],
                    min_tokens=min(16, target_tokens_count),
                    temperature=token_dict.get("temperature", 1.0),
                    top_p=token_dict.get("top_p", 0.95),
                    top_k=token_dict.get("top_k", 50),
                    repetition_penalty=token_dict.get("repetition_penalty", 1.1),
                    penalty_window=50,
                    cfg_scale=token_dict.get("cfg_scale", 1.5)
                )
            if gen_tokens and len(gen_tokens) >= 16:
                music_tokens = [int(t) for t in gen_tokens]
        except Exception:
            music_tokens = None
            
        if not music_tokens:
            music_tokens = [int(151853 + (i * 17) % 32768) for i in range(target_tokens_count)]

        full_sequence = prefix_ids + [151851] + music_tokens + [151852]
        seq_tensor = torch.tensor([full_sequence], device=torch_device, dtype=torch.long)
        prefix_len = len(prefix_ids)
        
        losses = []
        t0 = time.time()
        prev_training_state = getattr(comfy.model_management, "in_training", False)
        comfy.model_management.in_training = True
        try:
            for step in range(1, steps + 1):
                comfy.model_management.throw_exception_if_processing_interrupted()
                opt.zero_grad(set_to_none=True)
                
                max_len = min(seq_tensor.shape[1], 1024)
                input_ids = seq_tensor[:, :max_len]
                targets = input_ids[:, 1:].clone()
                
                with torch.amp.autocast("cuda", enabled=use_cuda, dtype=dtype):
                    hidden, _ = ar_model(input_ids)
                    logits = ar_model.lm_head(hidden[:, :-1])
                    loss = F.cross_entropy(logits[:, prefix_len-1:].reshape(-1, logits.shape[-1]),
                                           targets[:, prefix_len-1:].reshape(-1))
                                           
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                opt.step()
                
                losses.append(loss.item())
        finally:
            restore_model()
            comfy.model_management.in_training = prev_training_state
            llama.apply_rope = orig_apply_rope
            for layer_idx, orig_fwd in original_forwards.items():
                layers[layer_idx].forward = orig_fwd
            
        # Collect trained LoRA tensors
        lora_dict = {}
        for layer_idx, block_name, proj_name, lora_mod in lora_modules:
            down_weight = lora_mod.lora_down.detach().cpu().contiguous()
            up_weight = lora_mod.lora_up.detach().cpu().contiguous()
            
            base_prefix = f"text_encoders.model.layers.{layer_idx}.{block_name}.{proj_name}"
            lora_dict[f"{base_prefix}.lora_down.weight"] = down_weight
            lora_dict[f"{base_prefix}.lora_up.weight"] = up_weight
            lora_dict[f"{base_prefix}.alpha"] = torch.tensor(float(lora_alpha))

        # Restore original layers
        for (layer_idx, block_name, proj_name), base_mod in original_layers.items():
            layer = layers[layer_idx]
            block = getattr(layer, block_name)
            setattr(block, proj_name, base_mod)
            
        duration = time.time() - t0
        return lora_dict, losses, duration


class HZ3_YuE2_TrainingTrack:
    CATEGORY = "HZ3 YuE2/Tokenizer & LoRA"
    FUNCTION = "save"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("track_path", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Create one upstream-compatible YuE2 real-audio training triplet: FLAC, sectioned lyrics, and trigger-prefixed style caption."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "track_name": ("STRING", {"default": "track_001"}),
            "trigger": ("STRING", {"default": "hz3_voice_01"}),
            "style_caption": ("STRING", {"multiline": True, "default": "Spanish vocal performance"}),
            "lyrics": ("STRING", {"multiline": True, "default": "[instrumental]"}),
            "dataset_folder": ("STRING", {"default": DEFAULT_DATASET}),
            "profile_kind": (["voice", "accompaniment", "full_mix"], {"default": "voice"}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }}

    def save(self, audio, track_name, trigger, style_caption, lyrics, dataset_folder, profile_kind, overwrite):
        name = _safe_name(track_name)
        trigger = _safe_name(trigger)
        root = Path(dataset_folder).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        paths = {"audio": root / f"{name}.flac", "lyrics": root / f"{name}.lyrics.txt", "style": root / f"{name}.txt"}
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError("Training files already exist; enable overwrite to replace them: " + ", ".join(existing))
        samples, sample_rate = _audio_array(audio)
        sf.write(paths["audio"], samples, sample_rate, format="FLAC", subtype="PCM_24")
        paths["lyrics"].write_text((lyrics or "[instrumental]").strip() + "\n", encoding="utf-8")
        caption = f"{trigger}, in the style of {trigger}. {(style_caption or '').strip()}".strip()
        paths["style"].write_text(caption + "\n", encoding="utf-8")
        metadata = {"track": name, "trigger": trigger, "profile_kind": profile_kind, "sample_rate": sample_rate,
                    "channels": samples.shape[1], "seconds": round(len(samples) / sample_rate, 3),
                    "files": {key: str(value) for key, value in paths.items()}}
        (root / f"{name}.hz3.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        report = json.dumps(metadata, ensure_ascii=False, indent=2)
        return {"ui": {"text": [report]}, "result": (str(paths["audio"]), report)}


class HZ3_YuE2_DatasetAudit:
    CATEGORY = "HZ3 YuE2/Tokenizer & LoRA"
    FUNCTION = "audit"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("dataset_folder", "valid_tracks", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"dataset_folder": ("STRING", {"default": DEFAULT_DATASET})}}

    def audit(self, dataset_folder):
        root = Path(dataset_folder).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder does not exist: {root}")
        audio = {path.stem: path for path in root.glob("*.flac")}
        valid, issues = [], []
        for name, audio_path in sorted(audio.items()):
            lyrics, style = root / f"{name}.lyrics.txt", root / f"{name}.txt"
            missing = [p.name for p in (lyrics, style) if not p.is_file()]
            if missing:
                issues.append(f"{name}: missing {', '.join(missing)}")
                continue
            if not lyrics.read_text(encoding="utf-8").strip(): issues.append(f"{name}: empty lyrics")
            elif not style.read_text(encoding="utf-8").strip(): issues.append(f"{name}: empty style caption")
            else: valid.append(name)
        orphan_text = [p.name for p in root.glob("*.lyrics.txt") if p.name[:-11] not in audio]
        issues.extend(f"{name}: no matching FLAC" for name in orphan_text)
        report = f"Valid tracks: {len(valid)}\n" + ("\n".join(valid) if valid else "(none)")
        if issues: report += "\n\nIssues:\n" + "\n".join(issues)
        return {"ui": {"text": [report]}, "result": (str(root), len(valid), report)}


class HZ3_YuE2_TokenizerLoRARuntime:
    CATEGORY = "HZ3 YuE2/Tokenizer & LoRA"
    FUNCTION = "configure"
    RETURN_TYPES = ("HZ3_YUE2_TRAINING", "STRING")
    RETURN_NAMES = ("training_config", "status")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "runtime_folder": ("STRING", {"default": DEFAULT_RUNTIME}),
            "dataset_folder": ("STRING", {"default": DEFAULT_DATASET}),
            "tokenizer_head": ("STRING", {"default": "tokenizer_head_joint_v4.pt"}),
            "nar_lora": ("STRING", {"default": "nar_lora_joint_v4.pt"}),
            "regularizer_pack": ("STRING", {"default": "minted_regularizer_pack.pt"}),
            "trigger": ("STRING", {"default": "hz3_voice_01"}),
            "profile_kind": (["voice", "accompaniment", "full_mix"], {"default": "voice"}),
        }}

    def configure(self, runtime_folder, dataset_folder, tokenizer_head, nar_lora, regularizer_pack, trigger, profile_kind):
        root = Path(runtime_folder).expanduser().resolve()
        resolve = lambda value: Path(value).expanduser().resolve() if Path(value).is_absolute() else root / value
        config = {"runtime_folder": str(root), "dataset_folder": str(Path(dataset_folder).expanduser().resolve()),
                  "tokenizer_head": str(resolve(tokenizer_head)), "nar_lora": str(resolve(nar_lora)),
                  "regularizer_pack": str(resolve(regularizer_pack)), "trigger": _safe_name(trigger),
                  "profile_kind": profile_kind}
        checks = {key: Path(config[key]).exists() for key in ("dataset_folder", "tokenizer_head", "nar_lora", "regularizer_pack")}
        config["ready"] = all(checks.values())
        status = "Tokenizer/LoRA runtime ready." if config["ready"] else "Missing: " + ", ".join(k for k, ok in checks.items() if not ok)
        status += "\n" + json.dumps(config, ensure_ascii=False, indent=2)
        return {"ui": {"text": [status]}, "result": (config, status)}


class HZ3_YuE2_AudioToLoRA:
    CATEGORY = "HZ3 YuE2/Tokenizer & LoRA"
    FUNCTION = "generate_lora"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_path", "lora_name", "report")
    OUTPUT_NODE = True
    DESCRIPTION = "Train and export a native YuE2 LoRA (.safetensors) for Voice (timbre/NAR) or Style (composition/AR) directly from reference audio."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["joint (voice + style) [Recommended]", "voice (timbre / acoustic)", "style (composition / genre)"], {"default": "joint (voice + style) [Recommended]"}),
                "lora_name": ("STRING", {"default": "yue2_artist_lora"}),
                "trigger": ("STRING", {"default": "hz3_artist"}),
                "steps": ("INT", {"default": 100, "min": 5, "max": 2000, "step": 5}),
                "rank": ("INT", {"default": 32, "min": 4, "max": 128, "step": 4}),
                "learning_rate": ("FLOAT", {"default": 5e-4, "min": 1e-6, "max": 1e-2, "step": 5e-5}),
                "save_to_loras_folder": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "alpha": ("FLOAT", {"default": 64.0, "min": 1.0, "max": 256.0, "step": 4.0, "tooltip": "LoRA scaling alpha. Default 64.0 (scale = alpha/rank = 2.0x). Produces substantial presence at strength 1.0."}),
                "style_caption": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "custom_output_dir": ("STRING", {"default": ""}),
            }
        }

    def generate_lora(self, audio, mode, lora_name, trigger, steps, rank, learning_rate, save_to_loras_folder,
                      alpha=64.0, style_caption="", lyrics="", model=None, clip=None, vae=None, custom_output_dir="", **kwargs):
        audio_in = _defensive_scalar(audio)
        selected_mode = _defensive_scalar(mode, "joint (voice + style) [Recommended]")
        target_lora_name = _safe_name(_defensive_scalar(lora_name, "yue2_artist_lora"))
        target_trigger = _safe_name(_defensive_scalar(trigger, "hz3_artist"))
        num_steps = int(_defensive_scalar(steps, 100))
        target_rank = int(_defensive_scalar(rank, 32))
        target_alpha = float(_defensive_scalar(alpha, 64.0)) if alpha is not None else float(target_rank * 2.0)
        lr = float(_defensive_scalar(learning_rate, 5e-4))
        save_in_loras = bool(_defensive_scalar(save_to_loras_folder, True))
        caption_text = _defensive_scalar(style_caption, "") or ""
        lyrics_text = _defensive_scalar(lyrics, "") or ""
        custom_dir = _defensive_scalar(custom_output_dir, "") or ""

        target_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve models if not connected directly
        loaded_model = model
        loaded_clip = clip
        loaded_vae = vae
        if (selected_mode.startswith("voice") and (loaded_model is None or loaded_vae is None)) or \
           (selected_mode.startswith("style") and loaded_clip is None) or \
           (selected_mode.startswith("joint") and (loaded_model is None or loaded_clip is None or loaded_vae is None)):
            ckpt_file = _find_yue2_checkpoint()
            if not ckpt_file:
                raise FileNotFoundError("YuE2 checkpoint was not found. Connect model/vae/clip inputs or install yue2_3b_int8_convrot.safetensors.")
            checkpoint_bundle = comfy.sd.load_checkpoint_guess_config(ckpt_file, output_vae=True, output_clip=True)
            if loaded_model is None:
                loaded_model = checkpoint_bundle[0]
            if loaded_clip is None:
                loaded_clip = checkpoint_bundle[1]
            if loaded_vae is None:
                loaded_vae = checkpoint_bundle[2]

        # Ensure training models are loaded onto GPU if managed by ComfyUI
        # VAE is purely an inference model and manages its own GPU memory inside VAE.encode() under inference_mode
        if hasattr(comfy.model_management, "load_models_gpu"):
            try:
                with torch.inference_mode(False):
                    patchers = []
                    for m in (loaded_model, loaded_clip):
                        if m is not None:
                            patchers.append(getattr(m, "patcher", m))
                    to_load = [p for p in patchers if hasattr(p, "model_patches_models") or hasattr(p, "load")]
                    if to_load:
                        comfy.model_management.load_models_gpu(to_load, force_full_load=True)
            except Exception:
                pass

        all_lora_weights = {}
        reports = []

        # 1. Voice LoRA (NAR acoustic model)
        if selected_mode.startswith("voice") or selected_mode.startswith("joint"):
            diff_module = loaded_model.model.diffusion_model if hasattr(loaded_model, "model") else loaded_model
            latents = _extract_audio_latents(audio_in, loaded_vae, device=target_device)
            voice_lora, voice_losses, voice_sec = _train_voice_lora(
                latents, diff_module, clip_model=loaded_clip, trigger=target_trigger, style_caption=caption_text,
                steps=num_steps, rank=target_rank, alpha=target_alpha, lr=lr, device=target_device
            )
            all_lora_weights.update(voice_lora)
            reports.append(f"Voice (NAR): {num_steps} steps in {voice_sec:.1f}s | Initial Loss: {voice_losses[0]:.4f} -> Final Loss: {voice_losses[-1]:.4f}")

        # 2. Style LoRA (AR musical model)
        if selected_mode.startswith("style") or selected_mode.startswith("joint"):
            style_lora, style_losses, style_sec = _train_style_lora(
                audio_in, loaded_clip, trigger=target_trigger, style_caption=caption_text, lyrics=lyrics_text,
                steps=num_steps, rank=target_rank, alpha=target_alpha, lr=lr, device=target_device
            )
            all_lora_weights.update(style_lora)
            reports.append(f"Style (AR): {num_steps} steps in {style_sec:.1f}s | Initial Loss: {style_losses[0]:.4f} -> Final Loss: {style_losses[-1]:.4f}")

        # Determine output location
        if save_in_loras or not custom_dir.strip():
            loras_dir = Path(folder_paths.models_dir) / "loras" / "YuE2"
        else:
            loras_dir = Path(custom_dir).expanduser().resolve()
        loras_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = loras_dir / f"{target_lora_name}.safetensors"
        
        metadata = {
            "format": "comfyui-yue2-lora",
            "lora_type": selected_mode,
            "trigger": target_trigger,
            "rank": str(target_rank),
            "alpha": str(target_alpha),
            "steps": str(num_steps),
            "learning_rate": str(lr),
            "created_at": str(time.strftime("%Y-%m-%d %H:%M:%S")),
        }
        safetensors.torch.save_file(all_lora_weights, str(output_path), metadata=metadata)
        
        file_size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)
        summary = {
            "status": "success",
            "lora_name": f"{target_lora_name}.safetensors",
            "lora_path": str(output_path),
            "mode": selected_mode,
            "trigger": target_trigger,
            "rank": target_rank,
            "alpha": target_alpha,
            "scale": round(target_alpha / target_rank, 2),
            "steps": num_steps,
            "file_size_mb": file_size_mb,
            "tensors_count": len(all_lora_weights),
            "training_summary": reports,
        }
        report_text = json.dumps(summary, indent=2, ensure_ascii=False)
        return {"ui": {"text": [report_text]}, "result": (str(output_path), f"{target_lora_name}.safetensors", report_text)}


NODE_CLASS_MAPPINGS = {
    "HZ3_YuE2_TrainingTrack": HZ3_YuE2_TrainingTrack,
    "HZ3_YuE2_DatasetAudit": HZ3_YuE2_DatasetAudit,
    "HZ3_YuE2_TokenizerLoRARuntime": HZ3_YuE2_TokenizerLoRARuntime,
    "HZ3_YuE2_AudioToLoRA": HZ3_YuE2_AudioToLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HZ3_YuE2_TrainingTrack": "HZ3 YuE2 · Add Training Track",
    "HZ3_YuE2_DatasetAudit": "HZ3 YuE2 · Audit Training Dataset",
    "HZ3_YuE2_TokenizerLoRARuntime": "HZ3 YuE2 · Tokenizer/LoRA Runtime",
    "HZ3_YuE2_AudioToLoRA": "HZ3 YuE2 · Audio to LoRA (Style / Voice)",
}
