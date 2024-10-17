import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

def compute_combined_similarities(query_ids, base_ids, features_dict, labels_dict, topk=100, batch_size=100, ignore_self=True):
    # Ensure consistent ordering of IDs
    base_id_to_index = {id: idx for idx, id in enumerate(base_ids)}
    results = []

    # Prepare feature vectors for each modality
    query_vectors = {}
    base_vectors = {}
    for modality in features_dict:
        feature = features_dict[modality]
        query_vectors[modality] = np.array([feature[id] for id in query_ids])
        base_vectors[modality] = np.array([feature[id] for id in base_ids])

    for i in tqdm(range(0, len(query_ids), batch_size)):
        batch_ids = query_ids[i:i+batch_size]
        batch_vectors = {modality: vectors[i:i+batch_size] for modality, vectors in query_vectors.items()}
        
        # Compute similarity matrices for each modality
        sim_matrices = {}
        for modality in features_dict:
            sim_matrices[modality] = 1 - cdist(batch_vectors[modality], base_vectors[modality], metric='cosine')
        
        # Sum the similarity matrices
        combined_sim = sum(sim_matrices.values())
        
        for j, sim in enumerate(combined_sim):
            query_id = batch_ids[j]
            
            if ignore_self:
                self_index = base_id_to_index.get(query_id, -1)
                if self_index != -1 and self_index < len(sim):
                    sim[self_index] = -np.inf
            
            # Store results for labels 0 and 1
            results_0 = []
            results_1 = []
            
            for idx, similarity in sorted(enumerate(sim), key=lambda x: x[1], reverse=True):
                base_id = base_ids[idx]
                label = labels_dict.get(base_id, -1)
                
                if label == 0 and len(results_0) < topk:
                    results_0.append({"vid": base_id, "similarity": float(similarity)})
                elif label == 1 and len(results_1) < topk:
                    results_1.append({"vid": base_id, "similarity": float(similarity)})
                
                if len(results_0) >= topk and len(results_1) >= topk:
                    break
            
            results.append({
                'vid': query_id, 
                'similarities': [
                    {'vid': [r['vid'] for r in results_0], 'sim': [r['similarity'] for r in results_0]},
                    {'vid': [r['vid'] for r in results_1], 'sim': [r['similarity'] for r in results_1]}
                ]
            })
    
    return results

# Main processing code
datasets = ['MultiHateClip/zh', 'MultiHateClip/en', 'HateMM']

for dataset in datasets:
    dataset_path = Path('data') / dataset
    
    # Load features from three modalities
    feature_files = {
        'audio': 'fea_audio_modal_retrieval.pt',
        'vision': 'fea_vision_modal_retrieval.pt',
        'text': 'fea_text_modal_retrieval.pt'
    }
    features_dict = {}
    for modality, file_name in feature_files.items():
        feature = torch.load(dataset_path / 'fea' / file_name, weights_only=True)
        # Process features
        for vid, feat in feature.items():
            if len(feat.shape) != 1:
                feature[vid] = feat.mean(dim=0)
            feature[vid] = feature[vid].numpy()  # Convert to NumPy array for cdist
        features_dict[modality] = feature
    
    # Get common video IDs across all modalities
    common_vids = set.intersection(*(set(features_dict[modality].keys()) for modality in features_dict))
    
    # Load video IDs
    train_vids = pd.read_csv(dataset_path / 'vids/train.csv', header=None)[0].tolist()
    valid_vids = pd.read_csv(dataset_path / 'vids/valid.csv', header=None)[0].tolist()
    test_vids = pd.read_csv(dataset_path / 'vids/test.csv', header=None)[0].tolist()
    train_valid_vids = list(set(train_vids + valid_vids))
    all_vids = list(set(train_valid_vids + test_vids))
    
    # Filter video IDs to those present in all modalities
    query_ids = [vid for vid in all_vids if vid in common_vids]
    base_ids = [vid for vid in train_valid_vids if vid in common_vids]
    
    print(f'Processing {dataset}')
    print(f'Number of query videos: {len(query_ids)}')
    print(f'Number of base videos: {len(base_ids)}')
    
    # Load labels
    labels = pd.read_json(dataset_path / 'label.jsonl', lines=True)
    labels_dict = labels.set_index('vid')['label'].to_dict()
    
    # Compute combined similarities
    result = compute_combined_similarities(
        query_ids, base_ids, features_dict, labels_dict, topk=100
    )
    
    # Set output path
    output_path = dataset_path / 'retrieval' / 'all_modal.jsonl'
    
    # Save results
    with open(output_path, 'w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')
    