import runpod
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import base64
from io import BytesIO
from PIL import Image

# Carica il modello
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def handler(event):
    try:
        input_data = event['input']
        prompt = input_data.get('prompt', 'Describe this video.')
        video_frames = input_data.get('video_frames', [])
        
        frames = []
        for frame_b64 in video_frames:
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data))
            frames.append(img)
        
        inputs = processor(text=prompt, images=frames, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = processor.decode(output[0], skip_special_tokens=True)
        
        return {"output": response}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
