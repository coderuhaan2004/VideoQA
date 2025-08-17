import pickle
import numpy as np
from PIL import Image
import torch
import faiss
from tqdm import tqdm

def faiss_process(preprocessor, model, frames, transcriptions, temp_path=".", device="cpu"):
    model.to(device)
    model.eval()

    frame_embeddings = []
    metadata = []

    # Build a list of transcriptions for easy lookup
    transcription_segments = list(transcriptions.values())

    for sec_index, frame_path in enumerate(tqdm(frames, desc="Processing frames")):
        image = Image.open(frame_path).convert("RGB")
        inputs = preprocessor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            frame_embeddings.append(image_features.cpu().numpy())

        # assume frame index = second in video
        frame_time = sec_index  

        # find matching transcription segment
        matched_text = ""
        for seg in transcription_segments:
            if seg["start_sec"] <= frame_time < seg["end_sec"]:
                matched_text = seg["text"]
                break

        metadata.append({
            "frame_path": frame_path,
            "frame_time": frame_time,
            "transcription": matched_text
        })

    print("Metadata done!")

    # Stack embeddings
    frame_embeddings = np.vstack(frame_embeddings).astype("float32")

    # Initialize FAISS index
    embedding_dim = frame_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(frame_embeddings)
    print("FAISS initialized!")

    # Save FAISS index
    faiss_path = f"{temp_path}/faiss_index.bin"
    faiss.write_index(index, faiss_path)
    print(f"FAISS index written to {faiss_path}")

    # Save metadata
    meta_path = f"{temp_path}/metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata written to {meta_path}")
