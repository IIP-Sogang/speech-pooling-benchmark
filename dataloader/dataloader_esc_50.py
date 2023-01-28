import pandas as pd
import librosa
import os

from torch.utils.data import Dataset



class ESC_Dataset(Dataset):
    def __init__(self, dataset, fold, audio_path, eval_mode = False):
        self.df_dataset = pd.read_csv(dataset)
        self.fold = fold
        self.eval_mode = eval_mode
        self.audio_path = audio_path

        # set dataset using the fold
        self.df_dataset = self.df_dataset.loc[self.df_dataset['fold'] == self.fold] if eval_mode else self.df_dataset.loc[self.df_dataset['fold'] != self.fold]
        
        # reset index
        self.df_dataset.reset_index(drop = True, inplace=True)

    def __getitem__(self, index):
        name = self.df_dataset.loc[index, 'filename']
        audio, sr = librosa.load(os.path.join(self.audio_path, name), sr = 16000) # 
        y = self.df_dataset.loc[index, 'target']

        return audio, y

    def __len__(self):
        return len(self.df_dataset)





def training_dataset(dataset, fold, audio_path, eval_mode, **kwargs):
	print('Initialised Adam optimizer')

	return ESC_Dataset(dataset, fold, audio_path, eval_mode, **kwargs)


def test_dataset(dataset, fold, audio_path, eval_mode, **kwargs):
	print('Initialised Adam optimizer')

	return ESC_Dataset(dataset, fold, audio_path, eval_mode, **kwargs)