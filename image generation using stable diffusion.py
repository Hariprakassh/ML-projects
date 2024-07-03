import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

def generate_image(prompt, output_path):
    # Load the pre-trained model from Hugging Face
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Generate the image
    with autocast("cuda"):
        image = pipe(prompt).images[0]

    # Save the image
    image.save(output_path)

if __name__ == "__main__":
    prompt = "A fantasy landscape with mountains and a river"
    output_path = "generated_image.png"
    generate_image(prompt, output_path)
    print(f"Image saved to {output_path}")
