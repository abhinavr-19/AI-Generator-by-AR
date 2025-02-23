import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import imageio
import os

# Load Stable Diffusion Model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to(device)

# Function to Generate Images and Create Video
def generate_video(prompt, num_frames=10, fps=5):
    images = []
    
    for _ in range(num_frames):
        image = pipe(prompt).images[0]
        images.append(image)

    # Save as video
    video_path = "output.mp4"
    imageio.mimsave(video_path, images, fps=fps)
    
    return video_path

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 AI Text-to-Video Generator by AR")
    
    with gr.Row():
        prompt_input = gr.Textbox(label="Enter a Text Prompt")
        fps_slider = gr.Slider(minimum=1, maximum=10, value=5, label="Frames per Second")
    
    output_video = gr.Video(label="Generated Video")

    generate_btn = gr.Button("Generate Video")

    generate_btn.click(
        fn=generate_video,
        inputs=[prompt_input, fps_slider],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch(share=True)
