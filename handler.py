import runpod
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import base64
from io import BytesIO
from PIL import Image
import requests
import cv2
import numpy as np
import tempfile
import os

# Carica il modello
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def extract_frames_from_video(video_path, num_frames=8):
    """Estrae frame uniformemente dal video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError("Video vuoto o corrotto")
    
    # Prendi frame uniformemente distribuiti
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Converti BGR (OpenCV) a RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frames.append(img)
    
    cap.release()
    return frames

def handler(event):
    try:
        input_data = event['input']
        prompt = input_data.get('prompt', 'Describe this video.')
        max_new_tokens = input_data.get('max_new_tokens', 256)
        
        # Supporta sia video_url che video_frames
        video_url = input_data.get('video_url')
        video_frames_b64 = input_data.get('video_frames', [])
        
        if video_url:
            # Download video
            response = requests.get(video_url, timeout=60)
            response.raise_for_status()
            
            # Salva temporaneamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(response.content)
                video_path = tmp_file.name
            
            try:
                # Estrai frame
                frames = extract_frames_from_video(video_path, num_frames=8)
            finally:
                # Cleanup
                os.remove(video_path)
        
        elif video_frames_b64:
            # Frame già forniti in base64
            frames = []
            for frame_b64 in video_frames_b64:
                img_data = base64.b64decode(frame_b64)
                img = Image.open(BytesIO(img_data))
                frames.append(img)
        else:
            raise ValueError("Devi fornire 'video_url' o 'video_frames'")
        
        if not frames:
            raise ValueError("Nessun frame estratto dal video")
        
        # Processa con il modello
        inputs = processor(text=prompt, images=frames, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response_text = processor.decode(output[0], skip_special_tokens=True)
        
        return {
            "output": {
                "caption": response_text,
                "frames_processed": len(frames)
            }
        }
    
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
