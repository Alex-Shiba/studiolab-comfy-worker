"""ReCaVSR's pinned video template, with internal file bindings and defaults."""

from contract import ID_PATTERN
import os


def build_vsr_graph(clip, job_id, video):
    parts = video.split("/") if isinstance(video, str) else []
    if (len(parts) != 2 or parts[0] != job_id or not parts[1].endswith(".mp4")
            or not ID_PATTERN.fullmatch(parts[1][:-4])):
        raise ValueError("invalid internal video name")

    def node(kind, **inputs):
        return dict(class_type=kind, inputs=inputs)

    # Node 16 retains the worker's WS timing convention, node 20 its output lookup.
    # Provisioning selects fast for CC >= 8.0, exact for 7.5, and rejects older GPUs.
    kernels = os.environ.get("COMFY_VSR_KERNELS", "fast")
    if kernels not in ("fast", "exact"):
        raise ValueError("invalid internal vsr kernels")
    return {
        "1": node("LoadVideo", file=video),
        "3": node("ReCaVSRLoader", transformer="transformer_int8_convrot.safetensors"),
        "16": node("ReCaVSRUpscale", model=["3", 0], video=["1", 0], scale=float(clip.vsr.scale),
                   color_fix="none", seed=42, decoder="flashdecoder", kernels=kernels, kv_offload=False, encoder="auto"),
        "20": node("SaveVideo", video=["16", 0], filename_prefix=job_id + "/" + job_id,
                   **{"format": "mp4", "format.codec": "auto"}),
    }
