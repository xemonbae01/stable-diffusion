from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline

app = FastAPI()

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v-1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Request model for the API
class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

@app.post("/generate/")
async def generate_image(request: GenerateRequest):
    try:
        # Generate image
        image = pipe(
            request.prompt, 
            num_inference_steps=request.num_inference_steps, 
            guidance_scale=request.guidance_scale
        ).images[0]

        # Save the image to a temporary location
        image_path = "generated_image.png"
        image.save(image_path)

        # Return the image path or URL (this can be a public URL in a production app)
        return {"image_path": image_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
