import runpod
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import requests
import cv2
import numpy as np
import tempfile
import os

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def extract_frames_from_video(video_path, num_frames=8):
    """Estrae frame uniformemente dal video e restituisce numpy array"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError("Video vuoto o corrotto")
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    if not frames:
        raise ValueError("Nessun frame estratto")
    
    return np.stack(frames)

def handler(event):
    try:
        input_data = event['input']
        prompt_text = input_data.get('prompt', 'Describe this video.')
        max_new_tokens = input_data.get('max_new_tokens', 256)
        video_url = input_data.get('video_url')
        
        if not video_url:
            raise ValueError("Devi fornire 'video_url'")
        
        response = requests.get(video_url, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(response.content)
            video_path = tmp_file.name
        
        try:
            video_frames = extract_frames_from_video(video_path, num_frames=8)
        finally:
            os.remove(video_path)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "video"},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, videos=video_frames, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response_text = processor.decode(output[0], skip_special_tokens=True)
        
        return {
            "output": {
                "caption": response_text,
                "frames_processed": len(video_frames)
            }
        }
    
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
