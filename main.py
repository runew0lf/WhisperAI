import datetime
import sys
from typing import Dict

import torch
import whisper
from pytube import Stream, YouTube
from tqdm import tqdm


class CustomProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current = self.n

    def update(self, n: int):
        super().update(n)
        self.current += n
        print(f"\rProgress: {self.current}/{self.total} ({self.current / self.total * 100:.2f}%)", end="", flush=True)


class TqdmForPyTube(tqdm):
    def on_progress(self, stream: Stream, chunk: bytes, bytes_remaining: int):
        self.total = stream.filesize
        bytes_downloaded = self.total - bytes_remaining
        return self.update(bytes_downloaded - self.n)


def download_video(url: str, filename: str) -> None:
    with TqdmForPyTube(unit="bytes") as pbar:
        print("Downloading video...")
        yt = YouTube(url)
        yt.register_on_progress_callback(pbar.on_progress)
        yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").first().download(
            filename=filename
        )
        print("video downloaded")


def transcribe_video(filename: str) -> Dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("small.en", device=device)
    option = whisper.DecodingOptions(language="en")
    sys.modules["whisper.transcribe"].tqdm.tqdm = CustomProgressBar

    return model.transcribe(filename)


def save_transcription_to_vtt(transcription: Dict, save_target: str) -> None:
    with open(save_target, "w", encoding="utf-8") as file:
        file.write("WEBVTT\n\n")
        segments = transcription["segments"]

        for index, segment in enumerate(tqdm(segments, desc="Saving transcription")):
            file.write(f"{index + 1}\n")
            start_time = str(datetime.timedelta(seconds=segment["start"]))
            end_time = str(datetime.timedelta(seconds=segment["end"]))
            file.write(f"{start_time} --> {end_time}\n")
            file.write(segment["text"].strip() + "\n\n")


def main():
    video_url = "https://www.youtube.com/watch?v=NEPrGCaBtoI"
    video_filename = "video.mp4"
    vtt_filename = "video.vtt"

    download_video(video_url, video_filename)
    print("Transcribing video")
    transcription = transcribe_video(video_filename)
    save_transcription_to_vtt(transcription, vtt_filename)
    print("Done!")


if __name__ == "__main__":
    main()
