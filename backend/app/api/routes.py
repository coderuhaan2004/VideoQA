from fastapi import APIRouter, UploadFile, Request, File, Form, HTTPException
from app.services.clip_model import clip_model
from app.services.llava_api import query_llava, process_query
from app.core.memory import ChatMemory
from app.services.extract_frames import extract_frames
from app.services.faiss import faiss_process
from app.services.extract_audio import AudioExtractor
from app.services.transcription import get_vosk_model
from app.services.transcribe import Transcriber

router = APIRouter()
memory = ChatMemory()

from PIL import Image

router = APIRouter()


@router.post("/clip")
async def clip_embed_video(file: UploadFile = File(...)):
    """
    Saves video, extracts frames, embeds them using CLIP, and stores in FAISS.
    """
    try:
        # Step 1: Save uploaded video to disk
        temp_path = "input_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Step 2: Extract frames
        frames = extract_frames(temp_path, "video_frames")
        (preprocess, model) = clip_model()

        extractor = AudioExtractor()
        audio_path = extractor.extract_audio(temp_path)

        #transcriptions handle
        model_path = get_vosk_model()
        transciber = Transcriber(model_path)
        print("Using model:", model_path)
        transcriptions = transciber.transcribe(audio_path)["transcription"]

        print("Creating index.bin and metadata.pkl")
        faiss_process(preprocess, model, frames, transcriptions)

        return {"ready": True}
    except Exception as e:
        return {"ready": False, "error": str(e)}


@router.post("/llava")
async def llava_query(request: Request):
    """
    Accepts a question from JSON, retrieves relevant context from FAISS,
    and sends it to Segmind's LLaVA API.
    """
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        print(f"Received question: {question}")

        if not question:
            raise HTTPException(status_code=400, detail="Question is required.")

        # Step 1: Retrieve relevant context
        (preprocess, model) = clip_model()
        context_data = process_query(question, preprocess, model)
        print(f"Retrieved context: {context_data}")

        # Step 2: Query LLaVA
        answer = query_llava(context_data, question)
        # memory.add(question, answer)

        print(f"LLaVA answer: {answer}")

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get LLaVA response: {str(e)}")

