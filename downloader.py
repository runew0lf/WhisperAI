import logging
from typing import Callable
from pytube import Stream, YouTube
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TqdmForPyTube(tqdm):
    def on_progress(self, stream: Stream, chunk: bytes, bytes_remaining: int) -> None:
        self.total = stream.filesize
        bytes_downloaded = self.total - bytes_remaining
        self.update(bytes_downloaded - self.n)


def download_video(url: str, filename: str, on_progress_callback: Callable = None) -> None:
    logger.info("Downloading video...")
    with TqdmForPyTube(unit="bytes") as pbar:
        yt = YouTube(url)
        if on_progress_callback is not None:
            yt.register_on_progress_callback(on_progress_callback)
        yt.register_on_progress_callback(pbar.on_progress)
        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").first().download(
            filename=filename
        )
    logger.info("Video downloaded.")
