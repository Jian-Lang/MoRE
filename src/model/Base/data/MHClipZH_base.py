from torch.utils.data import Dataset
import pandas as pd


class MHClipZH_Dataset(Dataset):
    def __init__(self):
        super(MHClipZH_Dataset, self).__init__()

    def _get_data(self, fold: int, split: str, task: str):
        if task == 'ternary':
            return self._get_data_ternary(fold, split, task)
        data = f'data/MultiHateClip/zh/data/annotation/{split}.tsv'
        # Video_ID	Majority_Voting	Label	Target_Victim	Component	Duration
        data = pd.read_csv(data, sep='\t')
        if task == 'binary':
            replace_value = {
                'Normal': 0,
                'Offensive': 1,
                'Hateful': 1,        
            }
        elif task == 'ternary':
            replace_value = {
                'Normal': 0,
                'Offensive': 1,
                'Hateful': 2,
            }
        else:
            raise NotImplementedError(f"Invalid task: {task}")
        data['label'] = data['Majority_Voting'].replace(replace_value)
        return data

    def _get_data_ternary(self, fold: int, split: str, task: str):
        if split == 'train':
            data = f'data/MultiHateClip/zh/data/annotation/{split}.tsv'
            data = pd.read_csv(data, sep='\t')
        elif split == 'test':
            data_valid = 'data/MultiHateClip/zh/data/annotation/valid.tsv'
            data_test = 'data/MultiHateClip/zh/data/annotation/test.tsv'
            # merge data_valid and data_test
            data_valid = pd.read_csv(data_valid, sep='\t')
            data_test = pd.read_csv(data_test, sep='\t')
            data = pd.concat([data_valid, data_test])
        else:
            raise ValueError(f'Invalid split: {split}')
        
        replace_value = {
            'Normal': 0,
            'Offensive': 1,
            'Hateful': 2,
        }
        data['label'] = data['Majority_Voting'].replace(replace_value)
        return data
