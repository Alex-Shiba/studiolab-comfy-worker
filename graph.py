"""Native H3 API graphs with isolated, validated per-job accelerator profiles."""

from contract import ClipRequest, FPS, ID_PATTERN
from lab import PROFILES, validate_lab

DIFFUSION = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = PROFILES["pika6"].filename


REF_DIFFUSION = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF_LORA = PROFILES["ref8_768"].filename


def build_graph(clip: ClipRequest, job_id: str, first=None, last=None, refs=(), video=None):
    if not ID_PATTERN.fullmatch(job_id):
        raise ValueError("invalid internal job id")
    if clip.mode == "regen":
        if first is not None or last is not None or refs:
            raise ValueError("invalid internal regen bindings")
        return build_regen_graph(clip, job_id, video)
    if video is not None:
        raise ValueError("video requires regen")
    if (clip.mode == "frames") != (first is not None) or (last is not None and first is None):
        raise ValueError("invalid internal frame bindings")
    if len(refs) != len(clip.refs) or (clip.mode == "refs") != bool(refs):
        raise ValueError("invalid internal reference bindings")
    for name in (first, last, *refs):
        if name is not None:
            # Only server-generated names inside this job are accepted.
            parts = name.split("/")
            if (len(parts) != 2 or parts[0] != job_id or
                    not parts[1].endswith((".png", ".jpg")) or
                    not ID_PATTERN.fullmatch(parts[1].rsplit(".", 1)[0])):
                raise ValueError("invalid internal image name")

    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    width, height = clip.dimensions
    g = {
        "1": node("UNETLoader", unet_name=DIFFUSION, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=TEXT_ENCODER, type="minimax", device="default"),
        "3": node("VAELoader", vae_name=VIDEO_VAE),
        "6": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0], prompt=clip.prompt,
                  width=width, height=height, length=clip.frames),
        "10": node("LoraLoaderModelOnly", model=["1", 0], lora_name=LORA, strength_model=1.0),
        "12": node("BasicGuider", model=["10", 0], conditioning=["6", 0]),
        "13": node("RandomNoise", noise_seed=clip.seed),
        "14": node("KSamplerSelect", sampler_name="res_multistep"),
        "15": node("BasicScheduler", model=["10", 0], scheduler="simple", steps=8, denoise=1.0),
        "16": node("SamplerCustomAdvanced", noise=["13", 0], guider=["12", 0], sampler=["14", 0],
                   sigmas=["15", 0], latent_image=["6", 1]),
        "17": node("VAEDecode", samples=["16", 0], vae=["3", 0]),
        "19": node("CreateVideo", images=["17", 0], fps=FPS),
        "20": node("SaveVideo", video=["19", 0], filename_prefix=job_id + "/" + job_id,
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }
    for key, number, name in (("first_frame", "5", first), ("last_frame", "7", last)):
        if name is not None:
            g[number] = node("LoadImage", image=name)
            g["6"]["inputs"][key] = [number, 0]
    if refs:
        g["1"]["inputs"]["unet_name"] = REF_DIFFUSION
        g["4"] = node("VAELoader", vae_name=AUDIO_VAE)
        g["6"]["class_type"] = "MiniMaxH3ReferenceToVideo"
        g["6"]["inputs"].update(audio_vae=["4", 0], ref_image_size="match")
        for index, name in enumerate(refs):
            number = str(30 + index)
            g[number] = node("LoadImage", image=name)
            g["6"]["inputs"][f"ref_images.ref_image_{index}"] = [number, 0]
    if clip.audio:
        g["4"] = node("VAELoader", vae_name=AUDIO_VAE)
        g["18"] = node("VAEDecodeAudio", samples=["16", 0], vae=["4", 0])
        g["19"]["inputs"]["audio"] = ["18", 0]
    lab = clip.lab or validate_lab({}, clip.mode)
    g["10"]["inputs"]["lora_name"] = PROFILES[lab.lora].filename
    g["15"]["inputs"]["steps"] = lab.steps
    model = ["10", 0]
    g["21"] = node("MiniMaxH3SigmaShift", model=model, shift_video=lab.shift[0], shift_audio=lab.shift[1])
    model = ["21", 0]
    if lab.attention == "kitchen":
        g["22"] = node("ModelAttentionBackend", model=model, attention="comfy kitchen attention")
        model = ["22", 0]
    if lab.sparse == "sol":
        g["23"] = node("BlockSparseAttention", model=model, selection="sol-attn",
                       **{"selection.tau": 1.3}, start_percent=.2, end_percent=1.0,
                       dense_blocks="", min_tokens=12288, extra_tokens=256,
                       sink_conditioning="exact_kv_and_rows", verbose=False)
        model = ["23", 0]
    g["12"]["inputs"]["model"] = model
    g["15"]["inputs"]["model"] = model
    if lab.lora == "pika6":
        g["14"]["inputs"]["sampler_name"] = "lcm"
        g["15"] = node("ManualSigmas", sigmas="1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0")
    return g


UPSCALE = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
HYPERFLOW = "minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors"
SHARP = "minimax_h3_lms_v1.0_r64.safetensors"
IC_LORA = "minimax_h3_open_regenerate_2k_ic_comfy_bf16.safetensors"
REFINER = "open_h3_regenerate_2k_lora.safetensors"
CURVES = "sharp_curve_branches.safetensors"
FAST_IC = "fast_h3_vae_v2_24l.safetensors"
FAST_REFINE = "fast_h3_vae_noncausal_24l.safetensors"


def build_regen_graph(clip, job_id, video):
    from regen import enabled
    if not enabled():
        raise ValueError("regen is disabled")
    parts = video.split("/") if isinstance(video, str) else []
    if (len(parts) != 2 or parts[0] != job_id or not parts[1].endswith(".mp4")
            or not ID_PATTERN.fullmatch(parts[1][:-4])):
        raise ValueError("invalid internal video name")

    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    width, height = clip.dimensions
    canvas_height = 1088 if height == 1080 else height
    ic = clip.regen.method == "ic"
    g = {
        "1": node("UNETLoader", unet_name=REF_DIFFUSION if ic else DIFFUSION, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=TEXT_ENCODER, type="minimax", device="default"),
        "3": node("VAELoader", vae_name=VIDEO_VAE),
        "30": node("LoadVideo", file=video),
        "31": node("GetVideoComponents", video=["30", 0]),
        "25": node("VAEEncode", pixels=["6", 0], vae=["3", 0]),
        "12": node("BasicGuider", model=["21", 0], conditioning=["8", 0]),
        "13": node("RandomNoise", noise_seed=clip.seed),
        "14": node("KSamplerSelect", sampler_name="euler"),
        "16": node("SamplerCustomAdvanced", noise=["13", 0], guider=["12", 0], sampler=["14", 0],
                   sigmas=["15", 0], latent_image=["28", 0]),
        # The pack's tiled fallback calls VAE.decode directly; it needs a plain
        # video tensor, whereas the sampler returns a joint AV NestedTensor.
        "29": node("LTXVSeparateAVLatent", av_latent=["16", 0]),
        "17": node("OpenH3RegenFastVAEDecode", samples=["29", 0],
                   vae_name=FAST_IC if ic else FAST_REFINE, keep_loaded=False, vae=["5" if ic else "3", 0]),
        "18": node("OpenH3RegenFinish", images=["17", 0], frame_count=clip.frames,
                   crop="auto (1088 -> 1080, 2176 -> 2160)"),
        # Sound is packet-copied from the original MP4 after this graph finishes.
        "19": node("CreateVideo", images=["18", 0], fps=FPS),
        "20": node("SaveVideo", video=["19", 0], filename_prefix=job_id + "/" + job_id,
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }
    if ic:
        g.update({
            "5": node("LynnRealH3VAELoader", vae_name=VIDEO_VAE, num_layers=0,
                      tile_size=256, tile_overlap=64, compile_decoder=False),
            "6": node("OpenH3ICDraft", images=["31", 0], source_fps=["31", 2],
                      target_width=1344, target_height=768, frame_count=124, output_fps=24,
                      fit="cover", audio_mode="trim_pad", audio=["31", 1]),
            "7": node("OpenH3ICPrompt", caption=clip.prompt),
            "8": node("MiniMaxH3ReferenceToVideo", clip=["2", 0], vae=["3", 0], prompt=["7", 0],
                      width=width, height=canvas_height, length=124, ref_image_size="match",
                      **{"ref_videos.ref_video_0": ["6", 0]}),
            "10": node("LoraLoaderModelOnly", model=["1", 0], lora_name=HYPERFLOW, strength_model=1.0),
            "11": node("LoraLoaderModelOnly", model=["10", 0], lora_name=SHARP, strength_model=1.0),
            "24": node("LoraLoaderModelOnly", model=["11", 0], lora_name=IC_LORA, strength_model=1.0),
            "21": node("MiniMaxH3SigmaShift", model=["24", 0], shift_video=12.0, shift_audio=3.0),
            "26": node("LatentUpscaleModelLoader", model_name=UPSCALE),
            "27": node("MiniMaxH3LatentUpscaleWithModel", model=["26", 0], samples=["25", 0],
                       width=width, height=canvas_height),
            "28": node("OpenH3ICComposeLatent", video_latent=["27", 0], reference_latent=["8", 1], pin_audio=True),
            "15": node("OpenH3ICSigmas", sampler="up3", custom_sigmas=""),
        })
    else:
        g.update({
            "4": node("VAELoader", vae_name=AUDIO_VAE),
            "6": node("OpenH3RegenPrepareDraft", images=["31", 0], source_fps=["31", 2],
                      size="768p (1344x768)", max_frames=clip.frames),
            "8": node("CLIPTextEncode", clip=["2", 0], text=clip.prompt),
            "10": node("OpenH3RegenLora", model=["1", 0], lora_name=REFINER, curve_name=CURVES,
                       curve_table=REF_DIFFUSION, strength=1.0, curve_strength=1.0, lora_dtype="bf16"),
            "21": node("BlockSparseAttention", model=["10", 0], selection="sol-attn",
                       **{"selection.tau": 1.2}, start_percent=0.0, end_percent=.9,
                       dense_blocks="0,1,49", min_tokens=12288, extra_tokens=256,
                       sink_conditioning="exact_kv_and_rows", verbose=False),
            "27": node("OpenH3RegenUpscale", latent=["25", 0], model_name=UPSCALE,
                       resolution="1080p (1920x1080)" if clip.regen.resolution == "1080p" else "2k (2560x1440)",
                       precision="bf16", temporal_chunking=True),
            "28": node("OpenH3RegenRefineLatent", latent=["27", 0], audio_vae=["4", 0],
                       audio_mode="silence (reference)"),
            "15": node("OpenH3RegenSigmas", steps=3, start_sigma=clip.regen.start_sigma),
        })
    return g


REQUIRED_NODES = frozenset({
    "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo", "EmptyMiniMaxH3LatentAV", "MiniMaxH3SigmaShift",
    "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "LoraLoaderModelOnly",
    "BasicGuider", "RandomNoise", "KSamplerSelect", "BasicScheduler",
    "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
})
