import argparse
import cv2
import os
import torch

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
)
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only
from moviepy.editor import VideoFileClip, concatenate_videoclips
from openai import OpenAI


def get_model(quantized):
    if quantized:
        # Quantized currently not working.
        quantization = int8_weight_only
        text_encoder = T5EncoderModel.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        quantize_(text_encoder, quantization())
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        quantize_(transformer, quantization())
        vae = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16
        ).to("cuda")
        quantize_(vae, quantization())
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
        )

        pipe.enable_sequential_cpu_offload()

    return pipe


def get_last_frame(last_video_path, output_path):
    cap = cv2.VideoCapture(last_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cv2.imwrite(output_path, frame)
    cap.release()


def generate_video(prompt, img_path, path, fps=8):
    if img_path is None:
        p = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16
        )
        p.enable_model_cpu_offload()
        video = p(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda"),
        ).frames[0]
        del p
    else:
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


def generate_new_prompt(prompt_progression):
    # Adapted prompt from https://github.com/THUDM/CogView3/blob/main/prompt_optimize.py
    sys_prompt = """
    You are part of a team of bots that creates images . You work with an assistant bot that will draw anything you say.
    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output a video of a forest morning , as described.
    You will be prompted by people looking to create detailed , amazing short form videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
    The end goal is to create a long video, so the short videos should build on each other in order to for a longer form video.
    There are a few rules to follow :
    - Prompt should always be written in English, regardless of the input language. Please provide the prompts in English.
    - You will only ever output a single video description per user request.
    - Video descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting.
    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.
    - Make sure the subsequent descriptions build on top of each other, but do not go too rapidly into new ideas.
    - Do not provide the process and explanation, just return the modified English description . Image descriptions must be between 100-200 words. Extra words will be ignored.
    """

    history = [
        {
            "role": "system",
            "content": sys_prompt,
        },
    ] + prompt_progression

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )

    new_prompt = completion.choices[0].message.content

    return new_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="The text prompt for video generation")
    parser.add_argument("loop_size", type=str, help="The number of loops")
    parser.add_argument("directory", type=str, help="Output directory")
    parser.add_argument("-q", action="store_true", help="Run quantized model")
    parser.add_argument(
        "--llm", action="store_true", help="use a llm to create a more dynamic video"
    )
    parser.add_argument(
        "--fps", type=str, default=8, help="output video frames per second"
    )
    args = parser.parse_args()

    pipe = get_model(args.q)

    if not os.path.isdir(args.directory):
        os.mkdir(args.directory)

    video_paths = []
    prompt_progression = []
    prompt = args.prompt
    last_frame_path = None
    for i in range(int(args.loop_size)):
        video_path = f"{args.directory}/{i}.mp4"
        video_paths.append(video_path)

        # Skip video generation if it already exists.
        if os.path.isfile(video_path):
            continue
        generate_video(prompt, last_frame_path, video_path, float(args.fps))

        last_frame_path = f"{args.directory}/{i}.jpg"
        get_last_frame(video_path, last_frame_path)

        if args.llm:
            new_prompt = generate_new_prompt(prompt_progression)
            prompt_progression.append({"role": "user", "content": prompt})
            prompt_progression.append({"role": "assistant", "content": new_prompt})
            prompt = new_prompt
            # Only keep last 5 prompt progressions
            prompt_progression = prompt_progression[-10:]

    concat_videos(video_paths, f"{args.directory}/final.mp4")
