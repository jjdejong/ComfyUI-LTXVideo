"""Helpers for assembling the reference/prompt inputs used by looping workflows."""

import re

import torch

import comfy.utils
from comfy_api.latest import io


_SLOT_NUMBER = re.compile(r"(\d+)$")


def _slot_number(name: str) -> int:
    """Sort autogrow slots by their numeric suffix rather than lexically."""
    match = _SLOT_NUMBER.search(name)
    return int(match.group(1)) if match else -1


def _ordered_values(values: dict, prefix: str) -> list:
    return [
        value
        for name, value in sorted(
            values.items(), key=lambda item: _slot_number(item[0])
        )
        if name.startswith(prefix) and value is not None
    ]


def _batch_images(images: list[torch.Tensor]) -> torch.Tensor:
    """Batch IMAGE tensors with the same normalization as ComfyUI's batch node."""
    if not images:
        raise ValueError("Connect at least one reference image.")

    max_channels = max(image.shape[-1] for image in images)
    padded_images = [
        torch.nn.functional.pad(image, (0, max_channels - image.shape[-1]), value=1.0)
        if image.shape[-1] < max_channels
        else image
        for image in images
    ]

    first_image_shape = padded_images[0].shape
    resized_images = []
    for image in padded_images:
        if image.shape[1:] != first_image_shape[1:]:
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1),
                first_image_shape[2],
                first_image_shape[1],
                "bilinear",
                "center",
            ).movedim(1, -1)
        resized_images.append(image)

    return torch.cat(resized_images, dim=0)


class LTXVTileReferencePrompts(io.ComfyNode):
    """Collect reference images and their per-tile prompt snippets."""

    @classmethod
    def define_schema(cls):
        image_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"),
            prefix="image",
            min=1,
            max=64,
        )
        prompt_template = io.Autogrow.TemplatePrefix(
            io.String.Input("prompt", multiline=True, default=""),
            prefix="prompt",
            min=1,
            max=64,
        )

        return io.Schema(
            node_id="LTXVTileReferencePrompts",
            display_name="LTX Tile References + Prompts",
            category="Lightricks/looping",
            search_aliases=[
                "LTX tile references",
                "LTX tile prompts",
                "looping reference batch",
            ],
            description=(
                "Batch looping reference images and append each tile prompt "
                "snippet to the common global prompt."
            ),
            inputs=[
                io.String.Input(
                    "global_prompt",
                    display_name="Global Prompt",
                    optional=True,
                    multiline=True,
                    default="",
                    dynamic_prompts=True,
                    tooltip="Prompt prefix shared by every tile.",
                ),
                io.Autogrow.Input(
                    "images",
                    template=image_template,
                    tooltip="Connect one image per reference slot (image0, image1, ...).",
                ),
                io.Autogrow.Input(
                    "prompts",
                    template=prompt_template,
                    tooltip="Connect one snippet per tile slot (prompt0, prompt1, ...).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="reference_images",
                    tooltip="Batched reference images in slot order.",
                ),
                io.String.Output(
                    display_name="prompts",
                    tooltip="Global prompt plus snippets, separated by |.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: io.Autogrow.Type,
        prompts: io.Autogrow.Type,
        global_prompt: str = "",
    ) -> io.NodeOutput:
        global_prompt = global_prompt or ""
        image_values = _ordered_values(images, "image")
        prompt_values = _ordered_values(prompts or {}, "prompt")

        reference_images = _batch_images(image_values)
        if not prompt_values:
            prompt_values = [""]

        tile_prompts = "|".join(
            f"{global_prompt}{snippet}" for snippet in prompt_values
        )
        return io.NodeOutput(reference_images, tile_prompts)
