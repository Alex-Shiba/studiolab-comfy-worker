"""Per-request experimental profiles; no runtime/model imports."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Profile:
    filename: str
    steps: int
    video: float
    audio: float = 3.0
    allowed_steps: tuple = ()


PROFILES = {
    "turbo8_544": Profile("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", 8, 12, allowed_steps=(4, 8)),
    "turbo8_768": Profile("minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", 8, 6),
    "turbo4_768": Profile("minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 4, 6),
    "pika6": Profile("pika_h3_flex_fl2va_comfy_bf16.safetensors", 6, 12),
    "ref_v01": Profile("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", 4, 12),
    "ref8_768": Profile("minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", 8, 6),
}


@dataclass(frozen=True)
class Lab:
    attention: str
    sparse: str
    lora: str
    steps: int
    shift: tuple[float, float]

    def public(self):
        result = asdict(self)
        result["shift"] = dict(zip(("video", "audio"), self.shift))
        return result


def validate_lab(value, mode):
    if type(value) is not dict or value.keys() - {"attention", "sparse", "lora", "steps", "shift"}:
        raise ValueError("invalid lab keys")
    attention, sparse = value.get("attention", "default"), value.get("sparse", "off")
    allowed = ("ref_v01", "ref8_768") if mode == "refs" else ("turbo8_544", "turbo8_768", "turbo4_768", "pika6")
    lora = value.get("lora", "ref8_768" if mode == "refs" else "turbo8_544")
    if type(attention) is not str or attention not in ("default", "kitchen"):
        raise ValueError("invalid lab attention")
    if type(sparse) is not str or sparse not in ("off", "sol"):
        raise ValueError("invalid lab sparse")
    if type(lora) is not str or lora not in allowed:
        raise ValueError("lab lora does not support mode")
    profile = PROFILES[lora]
    steps = value.get("steps", profile.steps)
    if type(steps) is not int or steps not in (profile.allowed_steps or (profile.steps,)):
        raise ValueError("lab steps not supported by profile")
    shift = value.get("shift", {"video": profile.video, "audio": profile.audio})
    if "shift" in value and lora == "pika6":
        raise ValueError("pika6 uses fixed shifts and sigmas")
    if (type(shift) is not dict or set(shift) != {"video", "audio"}
            or any(type(n) not in (int, float) or not .01 <= n <= 100 or not math.isfinite(n) for n in shift.values())):
        raise ValueError("lab shift requires video and audio from 0.01 to 100")
    return Lab(attention, sparse, lora, steps, (float(shift["video"]), float(shift["audio"])))
