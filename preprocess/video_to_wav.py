import os
import subprocess

from tqdm import tqdm


def convert_mp4_to_wav(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all mp4 files
    mp4_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    # Used to store filenames that cannot be converted
    unconverted_files = []

    # Create a progress bar using tqdm
    for filename in tqdm(mp4_files, desc="Conversion Progress"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_folder, output_filename)

        # Check if it has already been converted
        if os.path.exists(output_path):
            continue

        # Build the FFmpeg command
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            input_path,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "-loglevel",
            "error",  # Reduce FFmpeg output
            output_path,
        ]

        try:
            # Execute the FFmpeg command
            subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"\nError occurred while converting {filename}: {e.stderr.decode()}")
            unconverted_files.append(filename)

    # Print the filenames that could not be converted
    if unconverted_files:
        print("\nThe following files could not be converted:")
        for file in unconverted_files:
            print(file)
    else:
        print("\nAll files have been successfully converted!")


# Usage example
input_folder = "data/MultiHateClip/zh/videos"
output_folder = "data/MultiHateClip/zh/audios"

convert_mp4_to_wav(input_folder, output_folder)
print("Conversion completed!")
