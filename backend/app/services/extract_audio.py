import os
import subprocess

class AudioExtractor:
    def __init__(self, ffmpeg_path="ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def extract_audio(self, video_path, output_path=None, format="mp3"):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Default output path
        if output_path is None:
            base, _ = os.path.splitext(video_path)
            output_path = f"{base}.{format}"

        # Build ffmpeg command
        command = [
            self.ffmpeg_path,
            "-y",                # overwrite if file exists
            "-i", video_path,    # input file
            "-vn",               # no video
            "-ac", "1",          # mono
            "-ar", "16000",      # 16 kHz sample rate (good for ASR)
            "-f", format,        # output format
            output_path
        ]

        # Run command
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")

        return output_path
