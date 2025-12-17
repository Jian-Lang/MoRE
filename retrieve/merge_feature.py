from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def merge_feature(dataset_dir: str):
    fea_path = Path(dataset_dir) / "fea"
    if "zh" in dataset_dir:
        mtype = "chinese"
    else:
        mtype = "uncased"
    text_modal_fea = [
        torch.load(fea_path / f"fea_ocr_bert-base-{mtype}.pt", weights_only=True),
    ]
    if "HateMM" not in dataset_dir:
        text_modal_fea.append(torch.load(fea_path / f"fea_title_bert-base-{mtype}.pt", weights_only=True))
    vision_modal_fea = [
        torch.load(fea_path / "fea_frames_16_back_google-vit-base-16-224.pt", weights_only=True),
        torch.load(fea_path / "fea_frames_16_front_google-vit-base-16-224.pt", weights_only=True),
        torch.load(fea_path / "fea_frames_16_google-vit-base-16-224.pt", weights_only=True),
        torch.load(fea_path / f"fea_caption_bert-base-{mtype}.pt", weights_only=True),
    ]
    audio_model_fea = [
        torch.load(fea_path / "fea_audio_mfcc.pt", weights_only=True),
        torch.load(fea_path / f"fea_transcript_bert-base-{mtype}.pt", weights_only=True),
    ]
    vid_path = Path(dataset_dir) / "vids.csv"
    vids = pd.read_csv(vid_path, header=None)[0].tolist()
    text_fea_dict = {}
    vision_fea_dict = {}
    audio_fea_dict = {}
    # iterate all fea and merge by model
    for vid in tqdm(vids):
        text_fea = []
        vision_fea = []
        audio_fea = []
        for fea_dir in text_modal_fea:
            fea = fea_dir[vid]
            if len(fea.shape) == 2:
                fea = torch.mean(fea, dim=0)
            text_fea.append(fea)
        for fea_dir in vision_modal_fea:
            fea = fea_dir[vid]
            if len(fea.shape) == 2:
                fea = torch.mean(fea, dim=0)
            vision_fea.append(fea)
        for fea_dir in audio_model_fea:
            if len(fea.shape) == 2:
                fea = torch.mean(fea, dim=0)
            audio_fea.append(fea)
        text_fea_dict[vid] = torch.concat(text_fea, dim=0)
        vision_fea_dict[vid] = torch.concat(vision_fea, dim=0)
        audio_fea_dict[vid] = torch.concat(audio_fea, dim=0)
        # save to file
        torch.save(text_fea_dict, fea_path / "fea_text_modal_retrieval.pt")
        torch.save(vision_fea_dict, fea_path / "fea_vision_modal_retrieval.pt")
        torch.save(audio_fea_dict, fea_path / "fea_audio_modal_retrieval.pt")


merge_feature("data/HateMM")
merge_feature("data/MultiHateClip/zh")
merge_feature("data/MultiHateClip/en")
