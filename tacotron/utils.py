import pandas as pd
import numpy as np
import re
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset
from torchaudio import transforms


class TacotronPreprocessor:
    def __init__(
            self,
            save_intonation: bool = True,
            replace_symbol: bool = True
    ):
        self.save_intonation = save_intonation
        self.replace_symbol = replace_symbol
        self.vocab = None
        self.original_texts = None
        self.normalized_texts = None

        pass

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub('[^А-я \!\?\.\,\-]+', ' ', text)
        text = text.strip()
        return text

    def create_vocabulary(self):
        all = []
        for i in self.normalized_texts:
            all += i
        self.vocab = np.unique(all)

    def fit(self, texts: list):
        self.normalized_texts = [self.normalize_text(x) for x in texts]
        self.create_vocabulary()
        del self.normalized_texts

    def transform_single_text(self, text):
        text = text.lower()
        text = re.sub('[^А-я \!\?\.\,\-]+', '', text)
        temp_res = []
        for letter in text:
            index = np.where(self.vocab == letter)[0][0]
            temp_res.append(index)
        return temp_res

    def transform_all_texts(self, texts):
        res = []
        for txt in texts:
            res.append(self.transform_single_text(txt))

        res = np.asarray(res)
        return res

class TTSDataset(Dataset):
    def __init__(self, data_path='../RUSLAN_text/metadata_RUSLAN_22200.csv', resample_rate=12000):
        super().__init__()
        self.resample_rate = resample_rate
        self.data_path = data_path
        self.dataset = pd.read_csv(data_path, sep='|', header=None)
        self.dataset.columns = ['path', 'text']
        self.preprocessor = TacotronPreprocessor()
        self.preprocessor.fit(self.dataset.text.tolist())

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item):
        temp_row = self.dataset.iloc[item]
        path = '../RUSLAN/'+temp_row.path+'.wav'
        text = temp_row.text
        text_norm = self.preprocessor.transform_single_text(text)
        text_norm = torch.tensor(text_norm)
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.squeeze(0)
        new_waveform = F.resample(waveform, sample_rate, self.resample_rate)
        return text_norm, new_waveform


def collate_fn(data):
    texts, audios = zip(*data)
    max_text_length = max([x.shape[0] for x in texts])
    max_audio_length = max([x.shape[0] for x in audios]) + 1

    new_texts = torch.zeros(len(texts), max_text_length)
    for i in range(len(texts)):
        temp_text = texts[i]
        curr_text_length = temp_text.shape[0]
        new_texts[i][:curr_text_length] = temp_text

    new_audios = torch.zeros(len(audios), max_audio_length)
    for i in range(len(audios)):
        temp_audio = audios[i]
        temp_audio_length = temp_audio.shape[0]
        new_audios[i][1:temp_audio_length+1] = temp_audio

    spectrogram_transform = transforms.Spectrogram(n_fft=2048, win_length=int(24000*0.05), hop_length=int(24000*0.0125))
    spectrogram = spectrogram_transform(new_audios)
    mel_transform = transforms.MelScale(n_mels=80, sample_rate=24000, n_stft=2048 // 2 + 1)
    new_mel = mel_transform(spectrogram)

    return new_texts.long(), new_audios, new_mel, spectrogram
