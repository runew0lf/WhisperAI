import logging
from downloader import download_video
from transcriber import transcribe_video, save_transcription_to_vtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    video_url = "https://www.youtube.com/watch?v=NEPrGCaBtoI"
    video_filename = "video.mp4"
    vtt_filename = "video.vtt"

    download_video(video_url, video_filename)
    transcription = transcribe_video(video_filename)
    save_transcription_to_vtt(transcription, vtt_filename)
    logging.info("Done!")


if __name__ == "__main__":
    main()
