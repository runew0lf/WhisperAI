import logging
import sys
import datetime
from typing import Dict
import torch
import whisper
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CustomProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current = self.n

    def update(self, n: int):
        super().update(n)
        self.current += n
        print(f"\rProgress: {self.current}/{self.total} ({self.current / self.total * 100:.2f}%)", end="", flush=True)


def transcribe_video(filename: str) -> Dict:
    logger.info("Transcribing video...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("small.en", device=device)
    option = whisper.DecodingOptions(language="en")
    sys.modules["whisper.transcribe"].tqdm.tqdm = CustomProgressBar

    transcription = model.transcribe(filename)
    logger.info("Video transcribed.")
    return transcription


def save_transcription_to_vtt(transcription: Dict, save_target: str) -> None:
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
