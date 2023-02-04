from typing import Tuple, List, Union

import torch
import torch.nn as nn

from transformers import AutoProcessor, AutoModelForPreTraining, Wav2Vec2Model


class Extractor(nn.Module):
    def __init__(self)->None:
        super().__init__()

    def extract(self, inputs)->torch.Tensor:
        pass

class Wav2VecExtractor(Extractor):
    def __init__(self):
        super().__init__()
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base") # Contains tokenizer & encoder(convs)
        self.normalizer = processor.feature_extractor
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")    

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.normalizer(inputs, sampling_rate=sr, return_tensors='pt')
        inputs['input_values'] = inputs['input_values'].to(device)
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        return outputs.hidden_states


def load_extractor(ext_type='wav2vec2'):
    if ext_type == 'wav2vec2':
        return Wav2VecExtractor()