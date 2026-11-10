"""Native H3 API graphs with isolated, validated per-job accelerator profiles."""

from contract import ClipRequest, FPS, ID_PATTERN
from lab import PROFILES, validate_lab

DIFFUSION = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = PROFILES["turbo4_768"].filename


REF_DIFFUSION = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF_LORA = PROFILES["ref8_768"].filename


def build_graph(clip: ClipRequest, job_id: str, first=None, last=None, refs=()):
    if not ID_PATTERN.fullmatch(job_id):
        raise ValueError("invalid internal job id")
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


REQUIRED_NODES = frozenset({
    "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo", "EmptyMiniMaxH3LatentAV", "MiniMaxH3SigmaShift",
    "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "LoraLoaderModelOnly",
    "BasicGuider", "RandomNoise", "KSamplerSelect", "BasicScheduler",
    "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
})
