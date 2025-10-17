import runpod
import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import requests
import av
import tempfile
import os

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

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
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            video_frames_np = read_video_pyav(container, indices)
            
            # Converti numpy array in torch tensor PRIMA del processor
            video_frames = torch.from_numpy(video_frames_np).float()
            
            container.close()
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
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
