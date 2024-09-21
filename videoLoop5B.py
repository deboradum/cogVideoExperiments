import argparse
import cv2
import os
import torch

from diffusers import (
    CogVideoXImageToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips
from openai import OpenAI


pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()


def get_last_frame(last_video_path, output_path):
    cap = cv2.VideoCapture(last_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cv2.imwrite(output_path, frame)
    cap.release()


def generate_video(prompt, img_path, path, fps=8):
    img = load_image(img_path)
    video = pipe(
        image=img,
        prompt=prompt,
        guidance_scale=6,
        use_dynamic_cfg=True,
        num_inference_steps=50,
    ).frames[0]
    export_to_video(video, path, fps=fps)


def concat_videos(paths, output_video_path):
    video_clips = [VideoFileClip(path) for path in paths]
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(output_video_path, codec="libx264")

    for clip in video_clips:
        clip.close()


def generate_new_prompt(old_prompts):
    sys_prompt = """I have multiple prompts used to generate short videos. In the end,
    I want to concatenate all these short videos to make a long video. The main
    goals of the long video is to show many things. Your task is to create a new
    prompt for the video to continue. Make sure the changes in the scene are
    gradual, but that the scene over time is changing. Make sure to only respond
    with the new prompt, do not say or tell me anything else."""

    prev_prompts = " - ".join(old_prompts)

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": prev_prompts,
            },
        ],
    )

    new_prompt = completion.choices[0].message.content

    return new_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="The text prompt for video generation")
    parser.add_argument("loop_size", type=str, help="The number of loops")
    parser.add_argument("directory", type=str, help="Output directory")
    parser.add_argument("--fps", type=str, default=8, help="output video frames per second")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        os.mkdir(args.directory)

    video_paths = []
    last_frame_path = None
    for i in range(int(args.loop_size)):
        video_path = f"{args.directory}/{i}.mp4"
        video_paths.append(video_path)

        # Skip video generation if it already exists.
        if os.path.isfile(video_path):
            continue
        generate_video(args.prompt, last_frame_path, video_path, float(args.fps))

        last_frame_path = f"{args.directory}/{i}.jpg"
        get_last_frame(video_path, last_frame_path)

    concat_videos(video_paths, f"{args.directory}/final.mp4")
