import os
import re
from typing import List

import cv2
import jieba
import numpy as np
import pandas as pd
from Levenshtein import ratio
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

orc_reader = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)


def extract_frames(video_path, fps=1):
    frames = []
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    video.release()
    return frames


def frame_similarity(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


def remove_similar_frames(frames, threshold=0.95):
    if not frames:
        return []

    unique_frames = [frames[0]]
    for i in range(1, len(frames)):
        if frame_similarity(frames[i], frames[i - 1]) < threshold:
            unique_frames.append(frames[i])
    return unique_frames


def ocr_frames(frames):
    texts = []
    for frame in frames:
        # convert frame to numpy array
        result = orc_reader.ocr(frame, cls=True)
        text = ""
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    text += line[1][0] + " "
        # if text is not blank str
        if text:
            texts.append(text.strip())
    return texts


def remove_duplicate_texts(texts: List[str], threshold: float = 0.9) -> List[str]:
    if not texts:
        return []

    def similarity(s1: str, s2: str) -> float:
        words1 = list(jieba.cut(s1))
        words2 = list(jieba.cut(s2))

        return ratio(" ".join(words1), " ".join(words2))

    unique_texts = [texts[0]]
    for i in range(1, len(texts)):
        if all(similarity(texts[i], t) < threshold for t in unique_texts):
            unique_texts.append(texts[i])

    return unique_texts


def clean_and_correct_text(text):
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

    words = text.split()
    corrected_words = []
    for word in words:
        if word.isupper():
            corrected_words.append(word)
        else:
            corrected_words.append(spell(word))

    corrected_text = " ".join(corrected_words)

    return corrected_text.strip()


def extract_text_from_video(video_path):
    frames = extract_frames(video_path)
    # print(f'Extracted {len(frames)} frames')

    unique_frames = remove_similar_frames(frames)
    # print(f'Removed {len(frames) - len(unique_frames)} similar frames')

    texts = ocr_frames(unique_frames)
    # print text
    # for text in texts:
    #     print(f'{text}')

    cleaned_texts = [clean_and_correct_text(text) for text in texts]

    # for text in cleaned_texts:
    #     print(f'{text}')

    unique_texts = remove_duplicate_texts(cleaned_texts)

    return unique_texts


src_dir = "data/MultiHateClip/zh/videos"
dst_file = "data/MultiHateClip/zh/ocr.jsonl"

if not os.path.exists(dst_file):
    dst_df = pd.DataFrame(columns=["vid", "ocr"])
    dst_df.to_json(dst_file, orient="records", lines=True)
else:
    dst_df = pd.read_json(dst_file, lines=True)

cur_ids = dst_df["vid"].values if len(dst_df) > 0 else []

for file in tqdm(os.listdir(src_dir)):
    audio_file = os.path.join(src_dir, file)
    audio_id = file.replace(".mp4", "")

    if audio_id in cur_ids:
        continue

    ocr = ""
    result = extract_text_from_video(audio_file)
    for text in result:
        if len(text) > 3:
            ocr += text + "\n"

    # caption = image_to_caption(video_frames)
    tmp_df = pd.DataFrame([{"vid": audio_id, "ocr": ocr}])
    dst_df = pd.concat([dst_df, tmp_df], ignore_index=True)
    dst_df.to_json(dst_file, orient="records", lines=True, force_ascii=False)
