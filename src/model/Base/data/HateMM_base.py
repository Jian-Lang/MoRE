from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class HateMM_Dataset(Dataset):
    def __init__(self):
        super(HateMM_Dataset, self).__init__()

    def _get_data(self, fold: int, split: str, task: str):
        data = pd.read_csv('data/HateMM/HateMM_annotation.csv')
        if task == 'binary':
            pass
        else:
            raise NotImplementedError(f"Invalid task: {task}")
        replace_vaule = {
            'Hate': 1,
            'Non Hate': 0,
        }
        data['label'] = data['label'].replace(replace_vaule)
        data['Video_ID'] = data['video_file_name'].str.split('.').str[0]
        data['vid'] = data['Video_ID']
        if fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(data, fold, split)
        elif fold in ['default']:
            data = self._get_default_data(data, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data

    def _get_default_data(self, data, split):
        vid_file = f'data/HateMM/vids/{split}.csv'
        vids = pd.read_csv(vid_file, header=None)[0].tolist()
        data = data[data['vid'].isin(vids)]
        return data

    def _get_fold_data(self, data, fold: int, split: str):
        train_size, val_size, test_size = 0.7, 0.1, 0.2
        seed = 2024
        target_column = 'label'
        data_split = {}
        X = data.drop(columns=[target_column])
        y = data[target_column]
        y = y.astype('category')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold = fold - 1
        for i, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
            if i == fold:
                train_val_data = data.iloc[train_val_idx]
                data_split['test'] = data.iloc[test_idx]
                
                data_split['train'], data_split['valid'] = train_test_split(
                    train_val_data, 
                    test_size=val_size/(1-test_size),
                    stratify=train_val_data[target_column], 
                    random_state=seed
                )
                break
        data = data_split[split]
        return data
        