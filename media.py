"""Decode story images and sample video frames for a vision model."""
import base64
from io import BytesIO
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

MEDIA_VERSION = "heic-video-sampling-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
VIDEO_SUFFIXES = {".mp4", ".mov"}


class MediaError(ValueError):
    pass


def jpeg_data_url(source):
    image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((2048, 2048))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def story_images(story, context=False):
    """Return one image, or sampled JPEG frames (target: 3; context: middle frame)."""
    path = story["media_path"]
    if path.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(path) as source:
            return [jpeg_data_url(source)]
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise MediaError("Unsupported story media format; manual review needed.")
    duration = float(story["metadata"].get("video_duration") or 0)
    # Avoid seeking exactly to the end of a clip. With no duration, use its start.
    times = [duration * 0.5] if context else [0, duration * 0.5, duration * 0.9]
    images = []
    for timestamp in dict.fromkeys(times):
        try:
            result = subprocess.run([
                imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin",
                "-ss", str(max(0, timestamp)), "-i", str(path),
                "-frames:v", "1", "-an", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
            ], capture_output=True, timeout=30, check=True)
            with Image.open(BytesIO(result.stdout)) as frame:
                images.append(jpeg_data_url(frame))
        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
            raise MediaError("Video frame extraction failed; manual review needed.") from exc
    return images
