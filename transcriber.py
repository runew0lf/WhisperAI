import logging
import sys
import datetime
from typing import Dict
import torch
import whisper
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CustomProgressBar(tqdm):
    """A custom progress bar for use with WhisperAI.

    This class extends the tqdm progress bar to provide a custom progress bar for use with WhisperAI.

    Attributes:
        current: The current progress of the task being tracked.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a new instance of the CustomProgressBar class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)
        self.current = self.n

    def update(self, n: int):
        """Updates the progress bar with the current progress.

        Args:
            n: The number of iterations to increment the progress bar by.

        Returns:
            None
        """
        super().update(n)
        self.current += n
        print(f"\rProgress: {self.current}/{self.total} ({self.current / self.total * 100:.2f}%)", end="", flush=True)


def transcribe_video(filename: str) -> Dict:
    """Transcribes a video using WhisperAI.

    This function transcribes a video using WhisperAI and returns the transcription as a dictionary.

    Args:
        filename: The name of the file containing the video to transcribe.

    Returns:
        A dictionary containing the transcription of the video.
    """
    logger.info("Transcribing video...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("small.en", device=device)
    option = whisper.DecodingOptions(language="en")
    sys.modules["whisper.transcribe"].tqdm.tqdm = CustomProgressBar

    transcription = model.transcribe(filename)
    logger.info("Video transcribed.")
    return transcription


def save_transcription_to_vtt(transcription: Dict, save_target: str) -> None:
    """Saves a transcription to a WebVTT file.

    This function saves a transcription to a WebVTT file.

    Args:
        transcription: The transcription to save.
        save_target: The name of the file to save the transcription to.

    Returns:
        None
    """
    logger.info("Saving transcription...")
    with open(save_target, "w", encoding="utf-8") as file:
        file.write("WEBVTT\n\n")
        segments = transcription["segments"]

        for index, segment in enumerate(tqdm(segments, desc="Saving transcription")):
            file.write(f"{index + 1}\n")
            start_time = str(datetime.timedelta(seconds=segment["start"]))
            end_time = str(datetime.timedelta(seconds=segment["end"]))
            file.write(f"{start_time} --> {end_time}\n")
            file.write(segment["text"].strip() + "\n\n")
    logger.info("Transcription saved.")
