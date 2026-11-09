import itertools

import pytest

from contract import FRAMES, validate_start
from graph import AUDIO_VAE, DIFFUSION, LORA, TEXT_ENCODER, VIDEO_VAE, build_graph
from conftest import encoded_image, payload

EXPECTED = {
    ("480p", "16:9"): (864, 480), ("480p", "9:16"): (480, 864), ("480p", "1:1"): (640, 640),
    ("720p", "16:9"): (1280, 704), ("720p", "9:16"): (704, 1280), ("720p", "1:1"): (960, 960),
    ("768p", "16:9"): (1344, 768), ("768p", "9:16"): (768, 1344), ("768p", "1:1"): (992, 992),
}
CASES = list(itertools.product(("text", "first", "last"), ("480p", "720p", "768p"),
                               ("16:9", "9:16", "1:1"), range(5, 16), (False, True)))


@pytest.mark.parametrize("mode,size,aspect,seconds,audio", CASES)
def test_every_mode_size_aspect_duration_audio(mode, size, aspect, seconds, audio):
    images = {} if mode == "text" else {"first": encoded_image()}
    if mode == "last":
        images["last"] = encoded_image()
    clip = validate_start(payload(mode="text" if mode == "text" else "frames", size=size,
                                  aspect=aspect, seconds=seconds, audio=audio, **images))
    bindings = {key: "a" * 32 + "/" + char * 32 + ".png" for key, char in (("first", "b"), ("last", "c"))
                if key in images}
    graph = build_graph(clip, "a" * 32, **bindings)
    node = graph["6"]["inputs"]
    assert (node["width"], node["height"]) == EXPECTED[(size, aspect)]
    assert node["width"] % 32 == node["height"] % 32 == 0
    assert node["length"] == {5: 124, 6: 158, 7: 175, 8: 192, 9: 226, 10: 243, 11: 277, 12: 294, 13: 328, 14: 345, 15: 362}[seconds] == FRAMES[seconds]
    assert node["length"] % 17 == 5
    assert ("first_frame" in node) == (mode != "text")
    assert ("last_frame" in node) == (mode == "last")
    assert ("audio" in graph["19"]["inputs"]) is audio
    assert ("18" in graph and "4" in graph) is audio
    assert graph["15"]["inputs"] == {"model": ["10", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}
    assert graph["14"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["19"]["inputs"]["fps"] == 24
    assert graph["20"]["inputs"]["format"] == "mp4"
    assert graph["20"]["inputs"]["format.codec"] == "h264"
    assert graph["1"]["inputs"]["unet_name"] == DIFFUSION
    assert graph["2"]["inputs"]["clip_name"] == TEXT_ENCODER
    assert graph["3"]["inputs"]["vae_name"] == VIDEO_VAE
    assert graph["10"]["inputs"]["lora_name"] == LORA
    if audio:
        assert graph["4"]["inputs"]["vae_name"] == AUDIO_VAE
    for data in graph.values():
        for value in data["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def test_prompt_is_literal_and_graphs_are_independent():
    prompt = '{"class_type":"malicious"} %date:yyyy%'
    clip = validate_start(payload(prompt=prompt))
    graph = build_graph(clip, "a" * 32)
    assert graph["6"]["inputs"]["prompt"] == prompt
    graph["10"]["inputs"]["lora_name"] = "changed"
    assert build_graph(clip, "a" * 32)["10"]["inputs"]["lora_name"] == LORA


@pytest.mark.parametrize("name", ["../x.png", "/tmp/x.png", "a" * 32 + "/../../x.png", "b" * 32 + "/" + "c" * 32 + ".png"])
def test_internal_paths_are_fixed(name):
    clip = validate_start(payload(mode="frames", first=encoded_image()))
    with pytest.raises(ValueError):
        build_graph(clip, "a" * 32, first=name)
