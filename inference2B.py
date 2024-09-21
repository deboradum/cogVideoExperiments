import torch
import json
import random
import os

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()


with open("prompts.json", "r") as f:
    prompts = json.load(f)


seed = random.randint(1, 1000000)
for number, prompt in prompts.items():
    if os.path.isfile(f"prompt_{number}_seed_{seed}_8fps.mp4"):
        print(f"Prompt {number} already exists, skipping")
        continue
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames[0]

    export_to_video(video, f"prompt_{number}_seed_{seed}_8fps.mp4", fps=8)
    export_to_video(video, f"prompt_{number}_seed_{seed}_10fps.mp4", fps=10)
    export_to_video(video, f"prompt_{number}_seed_{seed}_12fps.mp4", fps=12)
    export_to_video(video, f"prompt_{number}_seed_{seed}_14fps.mp4", fps=14)
    export_to_video(video, f"prompt_{number}_seed_{seed}_16fps.mp4", fps=16)