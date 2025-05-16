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
# dataset_dir = 'data/HateMM'
output_file = os.path.join(dataset_dir, 'fea/fea_text_all_bert-base-chinese.pt')


# model_id = 'google-bert/bert-base-uncased'
model_id = 'google-bert/bert-base-chinese'

# model = AutoModel.from_pretrained("google-bert/bert-base-chinese", device_map='cuda')
# processor = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = AutoModel.from_pretrained(model_id, device_map='cuda')
processor = AutoTokenizer.from_pretrained(model_id)


class MyDataset(Dataset):
    def __init__(self):
        vid_file = os.path.join(dataset_dir, 'vids.csv')
        # each line of vid_file a vid
        with open(vid_file, 'r') as f:
            self.vids = [line.strip() for line in f]
        ocr_file = os.path.join(dataset_dir, 'ocr.jsonl')
        trans_file = os.path.join(dataset_dir, 'speech.jsonl')
        title_file = os.path.join(dataset_dir, 'title.jsonl')
        self.ocr_df = pd.read_json(ocr_file, lines=True)
        self.trans_df = pd.read_json(trans_file, lines=True)
        self.title_df = pd.read_json(title_file, lines=True)

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]
        ocr = self.ocr_df[self.ocr_df['vid'] == vid]['ocr'].values[0]
        trans = self.trans_df[self.trans_df['vid'] == vid]['transcript'].values[0]
        title = self.title_df[self.title_df['vid'] == vid]['text'].values[0]
        text = f'{title}\n{trans}\n{ocr}'
        # text = f'{ocr}\n{trans}'
        return vid, text

def customed_collate_fn(batch):
    # preprocess
    # merge to one list
    vids, texts = zip(*batch)
    inputs = processor(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    return vids, inputs

save_dict = {}

dataloader = DataLoader(MyDataset(), batch_size=1, collate_fn=customed_collate_fn, num_workers=0, shuffle=True)
model.eval()
for batch in tqdm(dataloader):
    with torch.no_grad():
        vids, inputs = batch
        inputs = inputs.to('cuda')
        pooler_output = model(**inputs)['last_hidden_state'][:,0,:]
        pooler_output = pooler_output.detach().cpu()
        # process outputs
        for i, vid in enumerate(vids):
            save_dict[vid] = pooler_output[i]

# save_dict to pickle
# torch.save(save_dict, os.path.join(output_dir, 'bert_chinese_tensor_512_hid.pt'))
torch.save(save_dict, output_file)


    