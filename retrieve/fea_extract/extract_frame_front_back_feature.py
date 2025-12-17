import os

import numpy as np
import torch
from PIL import Image
from regex import F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

dataset_dir = "data/HateMM"
frames_path = "frames_16_back"
# frames_path = "frames_16_front"
output_file = os.path.join(dataset_dir, f"fea/fea_{frames_path}_google-vit-base-16-224.pt")
model_id = "google/vit-base-patch16-224"
model = AutoModel.from_pretrained(model_id, device_map="cuda")
processor = AutoProcessor.from_pretrained(model_id)


class MyDataset(Dataset):
    def __init__(self):
        vid_file = os.path.join(dataset_dir, "vids.csv")
        with open(vid_file, "r") as f:
            self.vids = [line.strip() for line in f]

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        frames = []
        for i in range(16):  # Get 16 frames
            frame_path = os.path.join(dataset_dir, frames_path, f"{vid}", f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert("RGB")
                frames.append(frame)
            else:
                # If frame doesn't exist, fill with black image
                # raise ValueError(f"Frame {frame_path} not found")
                frames.append(Image.new("RGB", (224, 224), color="black"))
        return vid, frames


def collate_fn(batch):
    vids, all_frames = zip(*batch)
    all_frames = [frame for frames in all_frames for frame in frames]
    processed_frames = processor(images=all_frames, return_tensors="pt", padding=True)
    return vids, processed_frames


dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8)
features = {}

with torch.no_grad():
    for vids, processed_frames in tqdm(dataloader):
        bs = len(vids)
        inputs = {k: v.to(model.device) for k, v in processed_frames.items()}
        outputs = model(**inputs)
        hidden_stats = outputs.last_hidden_state.view(bs, 16, -1, outputs.last_hidden_state.size(-1))
        # Get features for each video
        for i, vid in enumerate(vids):
            features[vid] = (
                hidden_stats[i][
                    :,
                    0,
                ]
                .detach()
                .cpu()
            )
            # print(f"Video {vid}: Feature shape {features[vid].shape}")
# Save features

torch.save(features, output_file)
