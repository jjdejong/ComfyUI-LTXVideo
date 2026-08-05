"""Helpers for assembling the reference/prompt inputs used by looping workflows."""

import math
import re

import torch

import comfy.utils
from comfy_api.latest import io


_MAX_RESOLUTION = 16384
_SLOT_NUMBER = re.compile(r"(\d+)$")


def _slot_number(name: str) -> int:
    """Sort autogrow slots by their numeric suffix rather than lexically."""
    match = _SLOT_NUMBER.search(name)
    return int(match.group(1)) if match else -1


def _slot_numbers(*values: dict | None) -> list[int]:
    """Return the slot numbers present in any of the autogrow groups."""
    numbers = set()
    for group in values:
        for name in group or {}:
            number = _slot_number(name)
            if number >= 0:
                numbers.add(number)
    return sorted(numbers)


def _slot_value(values: dict | None, prefix: str, number: int, default=None):
    """Read one matched value from an autogrow group."""
    if not values:
        return default
    return values.get(f"{prefix}{number}", default)


def _parse_reference_indices(value: str | None, frame_count: int) -> list[int] | None:
    """Parse an optional comma/whitespace-separated reference index list."""
    if not value or not value.strip():
        return None

    try:
        indices = [int(token) for token in re.split(r"[,\s]+", value.strip())]
    except ValueError as error:
        raise ValueError(
            "reference_indices_override must contain only integer frame indices."
        ) from error

    invalid = [index for index in indices if index < 0 or index >= frame_count]
    if invalid:
        raise ValueError(
            "reference_indices_override contains indices outside the generated "
            f"frame range 0..{frame_count - 1}: {invalid}"
        )

    return indices


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


def _aligned_dimension(value: float, multiple: int, minimum: int) -> int:
    """Round a positive dimension to a supported pixel multiple."""
    return max(minimum, round(float(value) / multiple) * multiple)


def _resize_image(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize the first image in a slot to the requested reference dimensions."""
    if image.ndim != 4 or image.shape[0] < 1:
        raise ValueError(
            "The selected reference image must be a non-empty IMAGE batch."
        )

    return (
        comfy.utils.common_upscale(
            image[:1].movedim(-1, 1),
            width,
            height,
            "lanczos",
            crop="disabled",
        )
        .movedim(1, -1)
        .clamp(0, 1)
    )


class LTXVTileReferencePrompts(io.ComfyNode):
    """Collect looping references, prompts, and the temporal reference schedule."""

    TIME_SCALE = 8
    MIN_REFERENCE_OFFSET = 0.35

    @classmethod
    def _aligned_frames(cls, seconds, frame_rate, minimum):
        frames = round(seconds * frame_rate / cls.TIME_SCALE) * cls.TIME_SCALE
        return max(minimum, frames)

    @classmethod
    def _calculate_schedule(
        cls,
        frame_rate,
        total_duration,
        tile_duration,
        overlap_duration,
        reference_offset,
    ):
        frame_count = max(
            cls.TIME_SCALE + 1,
            math.floor((total_duration * frame_rate - 1) / cls.TIME_SCALE)
            * cls.TIME_SCALE
            + 1,
        )
        tile_size = min(cls._aligned_frames(tile_duration, frame_rate, 24), 1000)
        overlap = cls._aligned_frames(overlap_duration, frame_rate, 16)
        overlap = min(overlap, 80, tile_size - cls.TIME_SCALE)
        reference_margin = cls._aligned_frames(
            max(reference_offset, cls.MIN_REFERENCE_OFFSET), frame_rate, 0
        )
        reference_margin = min(reference_margin, tile_size - cls.TIME_SCALE)

        latent_frames = ((frame_count - 1) // cls.TIME_SCALE) + 1
        latent_tile_size = tile_size // cls.TIME_SCALE
        latent_overlap = overlap // cls.TIME_SCALE
        latent_stride = latent_tile_size - latent_overlap
        tile_count = max(
            1, math.ceil((latent_frames - latent_overlap) / latent_stride)
        )

        final_index = ((frame_count - 1) // cls.TIME_SCALE) * cls.TIME_SCALE
        tile_stride = tile_size - overlap
        reference_indices = [0]
        for tile_index in range(tile_count):
            reference_index = min(
                tile_index * tile_stride + tile_size - reference_margin,
                final_index,
            )
            reference_index -= reference_index % cls.TIME_SCALE
            if reference_index not in reference_indices:
                reference_indices.append(reference_index)

        return frame_count, tile_size, overlap, reference_indices, tile_count

    @classmethod
    def define_schema(cls):
        image_template = io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"image{i}" for i in range(64)],
            min=1,
        )
        prompt_template = io.Autogrow.TemplateNames(
            io.String.Input("prompt", multiline=True, default=""),
            names=[f"prompt{i}" for i in range(64)],
            min=1,
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
                "Calculate the looping schedule, batch enabled reference images, "
                "append each tile prompt snippet to the common global prompt, and "
                "prepare the selected scene/identity reference and Stage 1 dimensions."
            ),
            inputs=[
                io.String.Input(
                    "global_prompt",
                    display_name="Global Prompt",
                    optional=True,
                    multiline=True,
                    default="",
                    dynamic_prompts=True,
                    force_input=True,
                    tooltip="Prompt prefix shared by every tile.",
                ),
                io.Float.Input(
                    "frame_rate",
                    default=24.0,
                    optional=True,
                    min=0.01,
                    max=240.0,
                    step=0.01,
                    tooltip="Video frame rate used to calculate the temporal schedule.",
                ),
                io.Float.Input(
                    "total_duration",
                    default=48.0,
                    optional=True,
                    min=0.1,
                    max=3600.0,
                    step=0.1,
                    tooltip="Total output duration in seconds.",
                ),
                io.Float.Input(
                    "tile_duration",
                    default=10.0,
                    optional=True,
                    min=0.1,
                    max=3600.0,
                    step=0.1,
                    tooltip="Temporal tile duration in seconds.",
                ),
                io.Float.Input(
                    "overlap_duration",
                    default=2.0,
                    optional=True,
                    min=0.1,
                    max=3600.0,
                    step=0.1,
                    tooltip="Temporal overlap between adjacent tiles in seconds.",
                ),
                io.Float.Input(
                    "reference_offset",
                    default=0.5,
                    optional=True,
                    min=cls.MIN_REFERENCE_OFFSET,
                    max=3600.0,
                    step=0.1,
                    tooltip="Reference position as seconds before the end of each tile.",
                ),
                io.String.Input(
                    "reference_indices_override",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional comma-separated frame indices, one per tile slot. "
                        "Leave blank to calculate positions from the timing fields."
                    ),
                ),
                io.Int.Input(
                    "target_height",
                    default=1088,
                    optional=True,
                    min=32,
                    max=_MAX_RESOLUTION,
                    step=32,
                    tooltip=(
                        "Final output height. The reference is resized to an aligned "
                        "final resolution; Stage 1 dimensions are derived at half "
                        "resolution."
                    ),
                ),
                io.Int.Input(
                    "reference_image_index",
                    default=0,
                    optional=True,
                    min=0,
                    max=63,
                    step=1,
                    tooltip=(
                        "Zero-based named image slot to use for Scene Anchor and Face ID "
                        "(for example, 2 selects image2)."
                    ),
                ),
                io.Autogrow.Input(
                    "images",
                    template=image_template,
                    optional=True,
                    tooltip=(
                        "Connect named image slots (image0, image1, ...); intermediate "
                        "slots may remain disconnected. "
                        "A missing or bypassed source disables that slot and leaves a gap "
                        "in the reference schedule."
                    ),
                ),
                io.Autogrow.Input(
                    "prompts",
                    template=prompt_template,
                    optional=True,
                    tooltip=(
                        "Connect named prompt slots (prompt0, prompt1, ...); intermediate "
                        "slots may remain empty. "
                        "A tile may omit its image and still use a custom snippet; "
                        "empty snippets emit the global prompt."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="reference_images",
                    tooltip="Batched enabled reference images in slot order.",
                ),
                io.String.Output(
                    display_name="prompts",
                    tooltip="Global prompt plus snippets, separated by |.",
                ),
                io.Int.Output(display_name="frame_count"),
                io.Int.Output(display_name="temporal_tile_size"),
                io.Int.Output(display_name="temporal_overlap"),
                io.String.Output(
                    display_name="reference_indices",
                    tooltip="Generated or overridden indices for the enabled reference images.",
                ),
                io.Int.Output(display_name="tile_count"),
                io.Float.Output(display_name="frame_rate"),
                io.Int.Output(display_name="stage_1_width"),
                io.Int.Output(display_name="stage_1_height"),
                io.Image.Output(
                    display_name="reference_image",
                    tooltip=(
                        "Selected reference image, resized to the aligned final "
                        "resolution for Scene Anchor and Face ID."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: io.Autogrow.Type,
        prompts: io.Autogrow.Type,
        global_prompt: str = "",
        frame_rate: float = 24.0,
        total_duration: float = 48.0,
        tile_duration: float = 10.0,
        overlap_duration: float = 2.0,
        reference_offset: float = 0.5,
        reference_indices_override: str = "",
        target_height: int = 1088,
        reference_image_index: int = 0,
    ) -> io.NodeOutput:
        (
            frame_count,
            tile_size,
            overlap,
            calculated_indices,
            tile_count,
        ) = cls._calculate_schedule(
            frame_rate,
            total_duration,
            tile_duration,
            overlap_duration,
            reference_offset,
        )
        schedule_indices = _parse_reference_indices(
            reference_indices_override, frame_count
        ) or calculated_indices

        global_prompt = global_prompt or ""
        image_slot_numbers = _slot_numbers(images)
        configured_image_slots = set(image_slot_numbers)
        image_slots = {
            number: _slot_value(images, "image", number)
            for number in image_slot_numbers
            if isinstance(_slot_value(images, "image", number), torch.Tensor)
        }

        try:
            selected_reference = image_slots[int(reference_image_index)]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "reference_image_index must select a connected image slot: "
                f"image{reference_image_index}."
            ) from error

        source_height, source_width = selected_reference.shape[1:3]
        if source_height < 1 or source_width < 1:
            raise ValueError("The selected reference image has invalid dimensions.")

        final_height = _aligned_dimension(target_height, 64, 64)
        final_width = _aligned_dimension(
            final_height * source_width / source_height,
            64,
            64,
        )
        stage_1_height = final_height // 2
        stage_1_width = final_width // 2
        reference_image = _resize_image(
            selected_reference,
            final_width,
            final_height,
        )

        reference_images_list = []
        reference_indices = []
        last_image = None
        highest_image_slot = max(configured_image_slots, default=-1)
        prompt_numbers = _slot_numbers(prompts)
        final_schedule_position = len(schedule_indices) - 1
        for number, calculated_index in enumerate(schedule_indices):
            image = image_slots.get(number)
            if image is not None:
                last_image = image
            elif (
                number < highest_image_slot
                or number in configured_image_slots
                or number in prompt_numbers
            ):
                # An unwired/bypassed slot is an intentional gap. Preserve its
                # schedule position instead of shifting later references left.
                continue
            elif last_image is not None and number < final_schedule_position:
                # Match the sampler fallback for a schedule longer than the
                # supplied image list, except at the final tile endpoint. The
                # endpoint remains free unless its image is explicitly supplied.
                image = last_image

            if image is not None:
                reference_images_list.append(image)
                reference_indices.append(calculated_index)

        highest_prompt_slot = max(prompt_numbers, default=-1)
        # Prompt slots define tile prompt positions independently of image
        # slots, including custom prompts for tiles without an image.
        prompt_count = min(
            len(schedule_indices),
            max(highest_image_slot, highest_prompt_slot) + 1,
        )
        prompt_values = [
            _slot_value(prompts, "prompt", number, "") or ""
            for number in range(prompt_count)
        ]

        reference_images = _batch_images(reference_images_list)
        if not prompt_values:
            prompt_values = [""]

        tile_prompts = "|".join(
            f"{global_prompt}{snippet}" for snippet in prompt_values
        )
        return io.NodeOutput(
            reference_images,
            tile_prompts,
            frame_count,
            tile_size,
            overlap,
            ", ".join(str(index) for index in reference_indices),
            tile_count,
            frame_rate,
            stage_1_width,
            stage_1_height,
            reference_image,
        )
