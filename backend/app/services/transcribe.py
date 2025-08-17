import subprocess, sys, os, json
from datetime import datetime
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2

class Transcriber():
    def __init__(self, model_path, window_size_sec=5, stride_sec=1):
        """
        window_size_sec: context window size (e.g., 5 seconds)
        stride_sec: step size (e.g., 1 second)
        """
        self.model = Model(model_path)
        self.window_size = window_size_sec
        self.stride = stride_sec

    def transcribe(self, filename):
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        ffmpeg_command = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "quiet",
            "-i", filename,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "-"
        ]

        transcription = {}
        start_time = datetime.now()

        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=10**8) as process:
            audio = process.stdout.read()

        # Convert window/stride to bytes
        window_bytes = self.window_size * BYTES_PER_SECOND
        stride_bytes = self.stride * BYTES_PER_SECOND

        total_len = len(audio)
        frame_index = 0

        # Slide across audio
        for start in range(0, total_len - window_bytes + 1, stride_bytes):
            window = audio[start:start + window_bytes]
            rec = KaldiRecognizer(self.model, SAMPLE_RATE)
            rec.SetWords(True)

            if rec.AcceptWaveform(window):
                result = json.loads(rec.Result())
                text = result.get("text", "")
            else:
                part = json.loads(rec.PartialResult())
                text = part.get("partial", "")

            transcription[frame_index] = {
                "start_sec": start // BYTES_PER_SECOND,
                "end_sec": (start + window_bytes) // BYTES_PER_SECOND,
                "text": text
            }
            frame_index += 1

        # Handle tail
        if total_len % stride_bytes != 0:
            tail = audio[-window_bytes:]
            if tail:
                rec = KaldiRecognizer(self.model, SAMPLE_RATE)
                rec.SetWords(True)
                if rec.AcceptWaveform(tail):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")
                else:
                    part = json.loads(rec.PartialResult())
                    text = part.get("partial", "")

                transcription[frame_index] = {
                    "start_sec": (total_len - window_bytes) // BYTES_PER_SECOND,
                    "end_sec": total_len // BYTES_PER_SECOND,
                    "text": text
                }

        end_time = datetime.now()
        time_elapsed = end_time - start_time

        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_time": str(time_elapsed),
            "window_size": self.window_size,
            "stride": self.stride,
            "transcription": transcription  # dict of {index: {start_sec, end_sec, text}}
        }