import os
import re
import argparse

from moviepy.editor import VideoFileClip, concatenate_videoclips


def concat_videos(paths, output_video_path):
    print(paths)
    video_clips = [VideoFileClip(path) for path in paths]
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(output_video_path, codec="libx264")

    for clip in video_clips:
        clip.close()


def get_paths(directory):
    files = os.listdir(directory)
    videos = [
        f for f in files if f.endswith(".mp4") and re.match(r"^\d+\.mp4$", f)
    ]
    videos.sort(key=lambda x: int(re.match(r"^(\d+)\.mp4$", x).group(1)))

    paths = [os.path.join(directory, f) for f in videos]

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()

    concat_videos(get_paths(args.dir), f"{args.dir}/final.mp4")
