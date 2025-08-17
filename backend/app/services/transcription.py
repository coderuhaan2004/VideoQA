from app.services.transcribe import Transcriber
import os
import urllib.request
import zipfile

def get_vosk_model(model_name="vosk-model-small-en-us-0.15", target_dir="models"):
    # Where the final unzipped model will live
    model_path = os.path.join(target_dir, model_name)

    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path

    # Download zip
    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = os.path.join(target_dir, f"{model_name}.zip")

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(zip_path)

    print(f"Model ready at {model_path}")
    return model_path
