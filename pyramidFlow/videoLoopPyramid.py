import argparse
import cv2
import os
import torch

from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import load_image, export_to_video
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only
from moviepy.editor import VideoFileClip, concatenate_videoclips
from openai import OpenAI

from huggingface_hub import snapshot_download


def get_model(quantized):
    model_path = 'pyramidFLow'
    snapshot_download("rain1011/pyramid-flow-sd3", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')

    model_dtype = "bf16"
    model = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype,
        model_variant="diffusion_transformer_768p",  # 'diffusion_transformer_384p'
    )

    model.vae.to("cuda")
    model.dit.to("cuda")
    model.text_encoder.to("cuda")
    model.vae.enable_tiling()

    return model


def get_last_frame(last_video_path, output_path):
    cap = cv2.VideoCapture(last_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cv2.imwrite(output_path, frame)
    cap.release()


def generate_video(prompt, img_path, path, fps=8):
    torch_dtype = torch.bfloat16
    if img_path is None:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            video = model.generate(
            prompt=prompt,
            num_inference_steps=[20, 20, 20],
            video_num_inference_steps=[10, 10, 10],
            height=768,
            width=1280,
            temp=16,                    # temp=16: 5s, temp=31: 10s
            guidance_scale=9.0,         # The guidance for the first frame
            video_guidance_scale=5.0,   # The guidance for the other video latent
            output_type="pil",
            save_memory=True,
        )
    else:
        img = (Image.open(img_path).convert("RGB").resize((1280, 768)))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            video = model.generate_i2v(
            prompt=prompt,
            input_image=img,
            num_inference_steps=[10, 10, 10],
            temp=16,
            video_guidance_scale=4.0,
            output_type="pil",
            save_memory=True,
        )
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
        "--fps", type=str, default=24, help="output video frames per second"
    )
    args = parser.parse_args()

    model = get_model(args.q)

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
