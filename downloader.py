import logging
from typing import Callable
from pytube import Stream, YouTube
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TqdmForPyTube(tqdm):
    """A custom tqdm progress bar for use with PyTube.

    This class extends the tqdm progress bar to provide a custom progress bar for use with PyTube.

    Attributes:
        total: The total size of the file being downloaded.
    """

    def on_progress(self, stream: Stream, chunk: bytes, bytes_remaining: int) -> None:
        """Updates the progress bar with the current download progress.

        Args:
            stream: The stream being downloaded.
            chunk: The most recent chunk of data received.
            bytes_remaining: The number of bytes remaining to be downloaded.

        Returns:
            None
        """
        self.total = stream.filesize
        bytes_downloaded = self.total - bytes_remaining
        self.update(bytes_downloaded - self.n)


def download_video(url: str, filename: str, on_progress_callback: Callable = None) -> None:
    """Downloads a video from YouTube.

    This function downloads a video from YouTube and saves it to the specified file.

    Args:
        url: The URL of the video to download.
        filename: The name of the file to save the video to.
        on_progress_callback: An optional callback function to be called as the download progresses.

    Returns:
        None
    """
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
