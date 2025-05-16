import librosa
from transformers import ViTImageProcessor, ViTModel, CLIPVisionModel, CLIPImageProcessor
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, ChineseCLIPImageProcessor, ChineseCLIPVisionModel, ChineseCLIPTextModel, ChineseCLIPFeatureExtractor
from transformers import BertModel, BertTokenizer,AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch


dataset_dir = 'data/MultiHateClip/zh'
output_file = os.path.join(dataset_dir, 'fea/fea_audio_mfcc.pt')


# model_id = "google-bert/bert-base-uncased"

# model = AutoModel.from_pretrained("google-bert/bert-base-chinese", device_map='cuda')
# processor = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
# model = AutoModel.from_pretrained(model_id, device_map='cuda')
# processor = AutoTokenizer.from_pretrained(model_id)


class MyDataset(Dataset):
    def __init__(self):
        vid_file = os.path.join(dataset_dir, 'vids.csv')
        # each line of vid_file a vid
        with open(vid_file, 'r') as f:
            self.vids = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]
        audio_file = os.path.join(dataset_dir, 'audios', f'{vid}.wav')
        return vid, audio_file

def customed_collate_fn(batch):
    # preprocess
    # merge to one list
    vids, audio_files = zip(*batch)
    return vids, audio_files

save_dict = {}

dataloader = DataLoader(MyDataset(), batch_size=1, collate_fn=customed_collate_fn, num_workers=0, shuffle=True)
# model.eval()
for batch in tqdm(dataloader):
    with torch.no_grad():
        vids, audio_files = batch
        vid, audio_file = vids[0], audio_files[0]
        if not os.path.exists(audio_file):
            pooler_output = torch.zeros(128)
        else:
            y, sr = librosa.load(audio_file, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
            mfccs = mfccs.T
            # to tensor and mean in time axis
            mfccs = torch.tensor(mfccs).mean(dim=0)
            # print(f'vid: {vid}, mfccs.shape: {mfccs.shape}')

            pooler_output = mfccs
            # process outputs
        save_dict[vid] = pooler_output

# save_dict to pickle
# torch.save(save_dict, os.path.join(output_dir, 'bert_chinese_tensor_512_hid.pt'))
torch.save(save_dict, output_file)


    