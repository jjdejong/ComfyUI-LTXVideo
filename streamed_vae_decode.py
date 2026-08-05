import os
import subprocess
import gc
import logging
import datetime
import json
import tempfile

import torch

import comfy.model_management
import folder_paths
from comfy.utils import ProgressBar
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .nodes_registry import comfy_node
from .tiled_vae_decode import compute_chunk_boundaries


@comfy_node(
    name="LTXVStreamedTiledVAEDecode",
    description="LTXV Streamed Tiled VAE Decode to Video",
    experimental=True,
)
class LTXVStreamedTiledVAEDecode:
    """Decode temporal windows directly to an FFmpeg pipe.

    Unlike image-returning VAE decode nodes, this node keeps only the temporal
    overlap tail in memory. It is intended for long videos where a full IMAGE
    batch would otherwise become the limiting allocation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "video/LTX/streamed"}),
                "tile_size": ("INT", {"default": 320, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_tile_length": ("INT", {"default": 8, "min": 2, "max": 1000}),
                "temporal_overlap": ("INT", {"default": 2, "min": 0, "max": 8}),
                "codec": (["h265", "h264"], {"default": "h265"}),
                "crf": ("INT", {"default": 22, "min": 0, "max": 51}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "decode_to_video"
    CATEGORY = "Lightricks/video"
    OUTPUT_NODE = True

    @staticmethod
    def _ffmpeg_path():
        from shutil import which

        ffmpeg = which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg is required for streamed video output but was not found in PATH.")
        return ffmpeg

    @staticmethod
    def _frame_bytes(images):
        images = images.detach().to(device="cpu", dtype=torch.float32)
        images = images.clamp_(0.0, 1.0).mul_(255.0).add_(0.5).to(torch.uint8)
        return images.contiguous().numpy().tobytes()

    @staticmethod
    def _blend_overlap(previous, current):
        if previous.shape != current.shape:
            raise RuntimeError(
                "Temporal overlap shape changed between decode windows: "
                f"{tuple(previous.shape)} != {tuple(current.shape)}"
            )
        weights = torch.linspace(
            0.0,
            1.0,
            previous.shape[0] + 2,
            dtype=previous.dtype,
        )[1:-1].view(-1, 1, 1, 1)
        return previous * (1.0 - weights) + current * weights

    @staticmethod
    def _write_ffmetadata(video_metadata):
        if not video_metadata:
            return None
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata = json.dumps(video_metadata)
        metadata = metadata.replace("\\", "\\\\")
        metadata = metadata.replace(";", "\\;")
        metadata = metadata.replace("#", "\\#")
        metadata = metadata.replace("=", "\\=")
        metadata = metadata.replace("\n", "\\\n")
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="ltx-streamed-metadata-",
            suffix=".txt",
            dir=folder_paths.get_temp_directory(),
            delete=False,
        ) as stream:
            stream.write(";FFMETADATA1\ncomment=" + metadata)
            return stream.name

    @staticmethod
    def _save_metadata_image(path, image, prompt, extra_pnginfo):
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])
        image = image.clamp(0.0, 1.0).mul(255.0).add(0.5).to(torch.uint8)
        Image.fromarray(image.numpy()).save(path, pnginfo=metadata, compress_level=4)

    def decode_to_video(
        self,
        vae,
        latents,
        frame_rate,
        filename_prefix,
        tile_size,
        overlap,
        temporal_tile_length,
        temporal_overlap,
        codec,
        crf,
        save_metadata,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        samples = latents["samples"]
        if samples.ndim != 5:
            raise ValueError("LTX streamed decode expects video samples shaped [B, C, T, H, W].")
        if samples.shape[0] != 1:
            raise ValueError("LTX streamed decode currently supports batch size 1.")
        if temporal_tile_length < temporal_overlap + 1:
            raise ValueError("temporal_tile_length must be greater than temporal_overlap.")

        # This node is a terminal decoder: no downstream node needs the
        # diffusion model.  Unloading here makes the memory boundary reliable
        # even when a workflow-level cleanup node is missing or bypassed.
        comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()
        logging.info(
            "LTX streamed VAE decode: models unloaded; free memory before VAE load: %s",
            f"{comfy.model_management.get_free_memory():,}",
        )

        temporal_scale, spatial_x_scale, spatial_y_scale = vae.downscale_index_formula
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        tile_x = max(1, tile_size // spatial_x_scale)
        tile_y = max(1, tile_size // spatial_y_scale)
        overlap_x = max(0, overlap // spatial_x_scale)
        overlap_y = max(0, overlap // spatial_y_scale)

        total_latent_frames = samples.shape[2]
        total_output_frames = 1 + (total_latent_frames - 1) * temporal_scale
        overlap_output_frames = temporal_overlap * temporal_scale
        progress = ProgressBar(total_output_frames)

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        extension = "mp4"
        video_name = f"{filename}_{counter:05}.{extension}"
        video_path = os.path.join(full_output_folder, video_name)
        thumbnail_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.png")

        video_metadata = {}
        if save_metadata:
            if prompt is not None:
                video_metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                video_metadata.update(extra_pnginfo)
        metadata_path = self._write_ffmetadata(video_metadata)

        ffmpeg = self._ffmpeg_path()
        encoder = "libx265" if codec == "h265" else "libx264"
        process = None
        frame_shape = None
        tail = None
        written_frames = 0
        first_frame = None

        def write_frames(frames):
            nonlocal written_frames, first_frame
            if frames.numel() == 0:
                return
            if first_frame is None:
                first_frame = frames[0].detach().cpu().clone()
            process.stdin.write(self._frame_bytes(frames))
            process.stdin.flush()
            written_frames += frames.shape[0]
            progress.update(frames.shape[0])

        try:
            chunk_start = 0
            while chunk_start < total_latent_frames:
                overlap_start, chunk_end = compute_chunk_boundaries(
                    chunk_start,
                    temporal_tile_length,
                    temporal_overlap,
                    total_latent_frames,
                )
                chunk = samples[:, :, overlap_start:chunk_end]
                logging.info(
                    "LTX streamed VAE decode: latent window %s:%s (%s frames)",
                    overlap_start,
                    chunk_end,
                    chunk.shape[2],
                )
                decoded = vae.decode_tiled(
                    chunk,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    overlap=(overlap_x + overlap_y) // 2,
                    tile_t=chunk.shape[2],
                    overlap_t=1,
                )
                if decoded.ndim == 5:
                    decoded = decoded.reshape(-1, *decoded.shape[-3:])
                if decoded.ndim != 4:
                    raise RuntimeError(f"Unexpected VAE output shape: {tuple(decoded.shape)}")
                decoded = decoded.to(device="cpu", dtype=torch.float32)

                if process is None:
                    frame_shape = decoded.shape[1:3]
                    height, width = frame_shape
                    command = [
                        ffmpeg,
                        "-y",
                        "-v",
                        "error",
                    ]
                    if metadata_path is not None:
                        command += ["-f", "ffmetadata", "-i", metadata_path]
                    command += [
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        str(frame_rate),
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        encoder,
                        "-crf",
                        str(crf),
                        "-pix_fmt",
                        "yuv420p",
                    ]
                    if metadata_path is not None:
                        command += ["-map_metadata", "0", "-metadata", "creation_time=now"]
                    command += [
                        video_path,
                    ]
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

                if chunk_start == 0:
                    candidate = decoded
                else:
                    if decoded.shape[0] <= overlap_output_frames:
                        raise RuntimeError("Decoded temporal window is shorter than the configured overlap.")
                    overlap_frames = decoded[1 : 1 + overlap_output_frames]
                    candidate = torch.cat(
                        (self._blend_overlap(tail, overlap_frames), decoded[1 + overlap_output_frames :]),
                        dim=0,
                    )

                is_last_chunk = chunk_end >= total_latent_frames
                if is_last_chunk:
                    write_frames(candidate)
                    tail = None
                else:
                    if candidate.shape[0] <= overlap_output_frames:
                        raise RuntimeError("Temporal tile is too short to retain its overlap tail.")
                    write_frames(candidate[:-overlap_output_frames])
                    tail = candidate[-overlap_output_frames:].clone()

                del decoded, chunk, candidate
                comfy.model_management.soft_empty_cache()
                chunk_start = chunk_end

            if written_frames != total_output_frames:
                raise RuntimeError(
                    f"Streamed decode wrote {written_frames} frames; expected {total_output_frames}."
                )
        finally:
            if process is not None:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr is not None else ""
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(f"FFmpeg video encode failed:\n{stderr}")
            if metadata_path is not None:
                os.unlink(metadata_path)

        if save_metadata and first_frame is not None:
            self._save_metadata_image(thumbnail_path, first_frame, prompt, extra_pnginfo)

        output_path = video_path
        if audio is not None:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            if waveform is None or sample_rate is None:
                raise ValueError("audio must contain waveform and sample_rate.")
            waveform = waveform.squeeze(0).transpose(0, 1).detach().cpu().to(torch.float32)
            audio_name = f"{filename}_{counter:05}-audio.{extension}"
            audio_path = os.path.join(full_output_folder, audio_name)
            mux_command = [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                video_path,
                "-ar",
                str(sample_rate),
                "-ac",
                str(waveform.shape[1]),
                "-f",
                "f32le",
                "-i",
                "-",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                audio_path,
            ]
            result = subprocess.run(
                mux_command,
                input=waveform.contiguous().numpy().tobytes(),
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg audio mux failed:\n{result.stderr.decode(errors='replace')}")
            output_path = audio_path

        relative_path = os.path.relpath(output_path, output_dir)
        return {
            "ui": {
                "gifs": [{
                    "filename": os.path.basename(output_path),
                    "subfolder": os.path.dirname(relative_path),
                    "type": "output",
                    "format": f"video/{codec}-mp4",
                }]
            },
            "result": (output_path,),
        }
