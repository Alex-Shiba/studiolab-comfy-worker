"""The fixed native H3 Turbo8 API graph, derived from the successful local probe."""

from contract import ClipRequest, FPS, ID_PATTERN

DIFFUSION = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def build_graph(clip: ClipRequest, job_id: str, first=None, last=None):
    if not ID_PATTERN.fullmatch(job_id):
        raise ValueError("invalid internal job id")
    if (clip.mode == "frames") != (first is not None) or (last is not None and first is None):
        raise ValueError("invalid internal frame bindings")
    for name in (first, last):
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
    if clip.audio:
        g["4"] = node("VAELoader", vae_name=AUDIO_VAE)
        g["18"] = node("VAEDecodeAudio", samples=["16", 0], vae=["4", 0])
        g["19"]["inputs"]["audio"] = ["18", 0]
    return g


REQUIRED_NODES = frozenset({
    "MiniMaxH3ImageToVideo", "EmptyMiniMaxH3LatentAV", "MiniMaxH3SigmaShift",
    "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "LoraLoaderModelOnly",
    "BasicGuider", "RandomNoise", "KSamplerSelect", "BasicScheduler",
    "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
})
