import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage import io
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision.transforms.functional import normalize
from tqdm import tqdm

# Load pretrained model
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list)-> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set input and output directories
input_dir = "data/MultiHateClip/zh/frames_16"
output_front_dir = "data/MultiHateClip/zh/frames_16_front"
output_back_dir = "data/MultiHateClip/zh/frames_16_back"

# Ensure output directories exist
os.makedirs(output_front_dir, exist_ok=True)
os.makedirs(output_back_dir, exist_ok=True)

# Get all video folders
video_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

# Create progress bar with tqdm
for video_name in tqdm(video_folders, desc="Processing video folders"):
    video_path = os.path.join(input_dir, video_name)

    # Create corresponding output folders for each video
    os.makedirs(os.path.join(output_front_dir, video_name), exist_ok=True)
    os.makedirs(os.path.join(output_back_dir, video_name), exist_ok=True)

    # Process 16 images in each video folder
    for i in range(16):
        frame_name = f"frame_{i:03d}.jpg"
        image_path = os.path.join(video_path, frame_name)
        front_output_path = os.path.join(output_front_dir, video_name, frame_name)
        back_output_path = os.path.join(output_back_dir, video_name, frame_name)

        # Check if output files already exist, skip if they do
        if os.path.exists(front_output_path) and os.path.exists(back_output_path):
            continue
        
        if os.path.exists(image_path):
            # Read and preprocess image
            orig_im = io.imread(image_path)
            orig_im_size = orig_im.shape[0:2]
            model_input_size = [1024, 1024]
            image = preprocess_image(orig_im, model_input_size).to(device)

            # Inference
            result = model(image)

            # Post-processing
            result_image = postprocess_image(result[0][0], orig_im_size)

            # Create foreground and background images
            pil_im = Image.fromarray(result_image)
            orig_image = Image.open(image_path).convert("RGB")

            # Create white background
            white_bg = Image.new("RGB", orig_image.size, (255, 255, 255))

            # Foreground image: Combine original image with mask, fill transparent parts with white
            no_bg_image = Image.composite(orig_image, white_bg, pil_im)

            # Background image: Combine original image with inverted mask, fill transparent parts with white
            bg_mask = Image.fromarray(255 - np.array(pil_im))
            bg_image = Image.composite(orig_image, white_bg, bg_mask)

            # Save foreground and background images in jpg format
            no_bg_image.save(front_output_path, "JPEG")
            bg_image.save(back_output_path, "JPEG")

print("Done!")