import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def extract_bert_features(dataset_dir, model_id, output_file):
    class MyDataset(Dataset):
        def __init__(self, dataset_dir):
            vid_file = os.path.join(dataset_dir, "vids.csv")
            with open(vid_file, "r") as f:
                self.vids = [line.strip() for line in f]
            text_file = os.path.join(dataset_dir, "speech.jsonl")
            self.text_df = pd.read_json(text_file, lines=True)

        def __len__(self):
            return len(self.vids)

        def __getitem__(self, index):
            vid = self.vids[index]
            text = self.text_df[self.text_df["vid"] == vid]["transcript"].values[0]
            return vid, text

    def customed_collate_fn(batch):
        vids, texts = zip(*batch)
        inputs = processor(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=512)
        return vids, inputs

    model = AutoModel.from_pretrained(model_id, device_map="cuda")
    processor = AutoTokenizer.from_pretrained(model_id)

    save_dict = {}

    dataloader = DataLoader(
        MyDataset(dataset_dir), batch_size=256, collate_fn=customed_collate_fn, num_workers=0, shuffle=False
    )
    model.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            vids, inputs = batch
            inputs = inputs.to("cuda")
            pooler_output = model(**inputs)["last_hidden_state"][:, 0, :]
            pooler_output = pooler_output.detach().cpu()
            for i, vid in enumerate(vids):
                save_dict[vid] = pooler_output[i]

    torch.save(save_dict, output_file)


extract_bert_features(
    "data/MultiHateClip/en",
    "google-bert/bert-base-uncased",
    "data/MultiHateClip/en/fea/fea_transcript_bert-base-uncased.pt",
)
extract_bert_features(
    "data/MultiHateClip/zh",
    "google-bert/bert-base-chinese",
    "data/MultiHateClip/zh/fea/fea_transcript_bert-base-chinese.pt",
)
extract_bert_features(
    "data/HateMM", "google-bert/bert-base-uncased", "data/HateMM/fea/fea_transcript_bert-base-uncased.pt"
)
