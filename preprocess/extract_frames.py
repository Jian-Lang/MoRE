import os
import sys
import subprocess
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback

def is_valid_video(video_path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path], capture_output=True, text=True, check=True)
        duration = float(result.stdout)
        return duration > 0
    except subprocess.CalledProcessError:
        return False
    except ValueError:
        return False

def extract_frame(video_path, timestamp, output_file):
    subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-i', video_path, '-ss', str(timestamp), '-frames:v', '1', output_file], check=True)

def process_video(video_path, output_folder, num_frames):
    if not is_valid_video(video_path):
        print(f"Error: Invalid video file: {video_path}")
        return
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)

    # Check if the output folder already contains the specified number of frames
    if os.path.exists(video_output_folder):
        existing_frames = glob.glob(os.path.join(video_output_folder, "frame_*.jpg"))
        if len(existing_frames) == num_frames:
            print(f"Skip video {video_name}: {num_frames} frames already exist")
            return

    # If the folder exists but the number of frames is incorrect, delete and recreate it
    if os.path.exists(video_output_folder):
        shutil.rmtree(video_output_folder)

    os.makedirs(video_output_folder, exist_ok=True)

    # Use ffprobe to get the video duration
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-i', video_path], capture_output=True, text=True)
    duration_line = [line for line in result.stdout.split('\n') if 'duration' in line][0]
    duration = float(duration_line.split('=')[1])

    if not duration or duration <= 0:
        print(f"Error: Invalid video duration {video_path}: {duration}")
        return

    interval = duration / num_frames

    def extract_frame_with_error_handling(args):
        video_path, timestamp, output_file = args
        try:
            extract_frame(video_path, timestamp, output_file)
        except Exception as e:
            print(f"Error occurred while processing frame {output_file}: {str(e)}")
            traceback.print_exc()

    with ThreadPoolExecutor() as executor:
        tasks = [(video_path, i * interval, os.path.join(video_output_folder, f"frame_{i:03d}.jpg")) for i in range(num_frames)]
        list(executor.map(extract_frame_with_error_handling, tasks))


def main():
    num_frames = 32
    input_folder = 'data/MultiHateClip/zh/videos'
    output_folder = 'data/MultiHateClip/zh/frames_{}'.format(num_frames)

    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    video_paths = glob.glob(os.path.join(input_folder, '*.mp4'))
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        process_video(video_path, output_folder, num_frames)

if __name__ == "__main__":
    main()