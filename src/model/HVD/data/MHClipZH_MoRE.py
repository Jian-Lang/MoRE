import pandas as pd
import torch
from transformers import AutoTokenizer
from pathlib import Path

from ...Base.data.MHClipZH_base import MHClipZH_Dataset


class MHClipZH_MoRE_Dataset(MHClipZH_Dataset):
    def __init__(self, fold: int, split: str, task: str, ablation='No', num_pos=30, num_neg=30, **kargs):
        super(MHClipZH_MoRE_Dataset, self).__init__()
        fea_path = Path('data/MultiHateClip/zh/fea')
        sim_path = Path('data/MultiHateClip/zh/retrieval')
        
        self.data = self._get_data(fold, split, task)
        # main feature
        self.mfcc_fea = torch.load(fea_path / 'fea_audio_mfcc.pt', weights_only=True)
        self.text_fea = torch.load(fea_path / 'fea_text_all_bert-base-chinese.pt', weights_only=True)
        self.frame_fea = torch.load(fea_path / 'fea_frames_16_google-vit-base-16-224.pt', weights_only=True)
        # similarity
        self.sim_all_sim = pd.read_json(sim_path / 'all_modal.jsonl', lines=True)

        self.ablation = ablation
        self.num_pos = num_pos
        self.num_neg = num_neg
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        label = item['label']
        vid = item['Video_ID']

        text_fea = self.text_fea[vid]
        vision_fea = self.frame_fea[vid]
        audio_fea = self.mfcc_fea[vid]
        
        num_pos = self.num_pos
        num_neg = self.num_neg
        
        all_sim_pos_vids = self.sim_all_sim[self.sim_all_sim['vid'] == vid].iloc[0]['similarities'][0]['vid'][:num_pos]
        all_sim_neg_vids = self.sim_all_sim[self.sim_all_sim['vid'] == vid].iloc[0]['similarities'][1]['vid'][:num_neg]

        text_sim_pos_fea = torch.stack([self.text_fea[key] for key in all_sim_pos_vids])
        text_sim_neg_fea = torch.stack([self.text_fea[key] for key in all_sim_neg_vids])
        vision_sim_pos_fea = torch.stack([self.frame_fea[key] for key in all_sim_pos_vids])
        vision_sim_neg_fea = torch.stack([self.frame_fea[key] for key in all_sim_neg_vids])
        audio_sim_pos_fea = torch.stack([self.mfcc_fea[key] for key in all_sim_pos_vids])
        audio_sim_neg_fea = torch.stack([self.mfcc_fea[key] for key in all_sim_neg_vids])

        return {
            'vid': vid,
            'label': torch.tensor(label),
            'text_fea': text_fea,
            'vision_fea': vision_fea,
            'audio_fea': audio_fea,
            'text_sim_pos_fea': text_sim_pos_fea,
            'text_sim_neg_fea': text_sim_neg_fea,
            'vision_sim_pos_fea': vision_sim_pos_fea,
            'vision_sim_neg_fea': vision_sim_neg_fea,
            'audio_sim_pos_fea': audio_sim_pos_fea,
            'audio_sim_neg_fea': audio_sim_neg_fea,
        }

class MHClipZH_MoRE_Collator:
    def __init__(self, **kargs):
        pass

    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        labels = [item['label'] for item in batch]
        text_fea = [item['text_fea'] for item in batch]
        vision_fea = [item['vision_fea'] for item in batch]
        audio_fea = [item['audio_fea'] for item in batch]
        vision_sim_pos_fea = [item['vision_sim_pos_fea'] for item in batch]
        vision_sim_neg_fea = [item['vision_sim_neg_fea'] for item in batch]
        text_sim_pos_fea = [item['text_sim_pos_fea'] for item in batch]
        text_sim_neg_fea = [item['text_sim_neg_fea'] for item in batch]
        audio_sim_pos_fea = [item['audio_sim_pos_fea'] for item in batch]
        audio_sim_neg_fea = [item['audio_sim_neg_fea'] for item in batch]
        
        return {
            'vids': vids,
            'labels': torch.stack(labels),
            'text_fea': torch.stack(text_fea),
            'vision_fea': torch.stack(vision_fea),
            'audio_fea': torch.stack(audio_fea),
            'text_sim_pos_fea': torch.stack(text_sim_pos_fea),
            'text_sim_neg_fea': torch.stack(text_sim_neg_fea),
            'vision_sim_pos_fea': torch.stack(vision_sim_pos_fea),
            'vision_sim_neg_fea': torch.stack(vision_sim_neg_fea),
            'audio_sim_pos_fea': torch.stack(audio_sim_pos_fea),
            'audio_sim_neg_fea': torch.stack(audio_sim_neg_fea),
        }