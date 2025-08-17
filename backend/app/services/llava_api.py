import os
import requests
import logging
import torch
import faiss
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import base64
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("URL")
api_key = os.getenv("API_KEY")

# Use this function to convert an image file from the filesystem to base64
def image_file_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def query_llava(context: dict, question: str) -> str:
    try:
        data = {
        "images": image_file_to_base64(context["frame_path"]),
        "prompt": f"Question:{question}, Transcription for the frame: {context['transcription']}"
        }

        headers = {'x-api-key': api_key}
        response = requests.post(url, json=data, headers=headers)

        res_json = response.json()
        # print(res_json)

        # If Segmind uses "response" instead of "message"
        return res_json.get("response", "No answer found.")
    
    except requests.RequestException as e:
        logging.error(f"[Segmind LLaVA API Error] {e}")
        return "Sorry, couldn't contact the LLaVA API."


def process_query(user_prompt, clip_processor, clip_model):
    try:
        device = "cpu"
        print("\n=== Starting Query Processing ===")
        print(f"Processing query: {user_prompt}")

        # Load FAISS index and metadata
        print("Step 1: Loading FAISS index and metadata...")
        index = faiss.read_index("faiss_index.bin")
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        print(f"Loaded index with {index.ntotal} frames")

        # Encode the user prompt using CLIP
        print("Step 2: Encoding user prompt with CLIP...")
        inputs = clip_processor(text=[user_prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs).cpu().numpy()
        print(f"Prompt encoded successfully. Embedding shape: {text_embedding.shape}")

        # Find the most similar frame using cosine similarity
        print("Step 3: Finding most similar frame...")
        similarities = cosine_similarity(text_embedding, index.reconstruct_n(0, index.ntotal))
        best_frame_idx = similarities.argmax()

        # Get metadata for the best frame
        best_metadata = metadata[best_frame_idx]
        frame_path = best_metadata['frame_path']
        transcription = best_metadata['transcription']
        print(f"Best matching frame found: {frame_path}")
        print(f"Transcription: {transcription}")


        return {
            'frame_path': frame_path,
            'transcription': transcription,
            'similarity_score': similarities[0][best_frame_idx]
        }

    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise


from transformers import CLIPProcessor, CLIPModel

class LLAVAChatModel:
    def __init__(self, llava_processor, llava_model, clip_processor, clip_model, device="cuda"):
        print(f"Initializing LLAVAChatModel on device: {device}")
        self.device = device

        # Load LLAVA
        print("Loading LLAVA processor...")
        self.llava_processor = llava_processor

        print("Loading LLAVA model...")
        self.llava_model = llava_model.to(self.device)

        # Load CLIP
        print("Loading CLIP processor...")
        self.clip_processor = clip_processor

        print("Loading CLIP model...")
        self.clip_model = clip_model.to(self.device)

        print("Model initialization complete!")



    def generate_response(self, frame_path, user_prompt, chat_history, transcription):
        """Generate a response using LLAVA."""
        print("Loading and processing frame...")
        image = Image.open(frame_path)
        print("Formatting context...")
        formatted_history = "\n".join(
            f"{message['role'].capitalize()}: {message['content']}" for message in chat_history
        )
        prompt = f"""
        Chat History:
        {formatted_history}

        Frame Transcription:
        {transcription}

        Question: {user_prompt}
        Answer:"""
        print("Processing inputs for LLAVA...")
        inputs = self.llava_processor(
            text=[prompt], images=image, return_tensors="pt", padding=True
        ).to(self.device)
        print("Generating response...")
        output = self.llava_model.generate(**inputs, max_new_tokens=200)
        response = self.llava_processor.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
