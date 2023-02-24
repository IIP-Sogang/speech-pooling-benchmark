from typing import Tuple, List, Union

import torch
import torch.nn as nn

from transformers import AutoProcessor, AutoModelForPreTraining, Wav2Vec2Model, AutoFeatureExtractor, AutoModel


class Extractor(nn.Module):
    def __init__(self)->None:
        super().__init__()

    def extract(self, inputs)->torch.Tensor:
        pass

class Wav2VecExtractor(Extractor):
    # https://huggingface.co/facebook/wav2vec2-base
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


class VQWav2VecExtractor(Extractor):
    # https://huggingface.co/facebook/wav2vec2-base
    def __init__(self):
        super().__init__()
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base") # Contains tokenizer & encoder(convs)
        self.normalizer = processor.feature_extractor
        self.model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")    

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.normalizer(inputs, sampling_rate=sr, return_tensors='pt')
        inputs = inputs['input_values'].to(device)

        conv_features = self._conv_feature(inputs[0])
        quantized_features = self._quantize(conv_features)

        return quantized_features

    def _quantize(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_groups = self.model.quantizer.num_groups

        hidden_states = self.model.quantizer.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * num_groups, -1)
        codevector_idx = hidden_states.argmax(dim=-1)

        return codevector_idx.reshape(-1, num_groups)

    def _conv_feature(self, sig):
        feats = self.model.wav2vec2.feature_extractor(sig)
        feats = feats.transpose(1, 2)
        _, feats = self.model.wav2vec2.feature_projection(feats)
        return feats


# XLS-R
class Wav2VecXLSR03BExtractor(Extractor):
    # https://huggingface.co/facebook/wav2vec2-xls-r-300m
    def __init__(self):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m") # encoder(convs)
        self.model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt') # normalization
        inputs['input_values'] = inputs['input_values'].to(device)
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        return outputs.hidden_states


class VQWav2VecXLSR03BExtractor(Extractor):
    # https://huggingface.co/facebook/wav2vec2-xls-r-300m
    def __init__(self):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m") # encoder(convs)
        self.model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")    

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt') # normalization
        inputs = inputs['input_values'].to(device)

        conv_features = self._conv_feature(inputs[0])
        quantized_features = self._quantize(conv_features)

        return quantized_features

    def _quantize(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_groups = self.model.quantizer.num_groups

        hidden_states = self.model.quantizer.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * num_groups, -1)
        codevector_idx = hidden_states.argmax(dim=-1)

        return codevector_idx.reshape(-1, num_groups)

    def _conv_feature(self, sig):
        feats = self.model.wav2vec2.feature_extractor(sig)
        feats = feats.transpose(1, 2)
        _, feats = self.model.wav2vec2.feature_projection(feats)
        return feats



# Hubert-Large
class HubertLarge(Extractor):
    # https://huggingface.co/facebook/hubert-large-ll60k
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ll60k") # encoder(convs)
        self.model = AutoModel.from_pretrained("facebook/hubert-large-ll60k")    

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt') # normalization
        inputs['input_values'] = inputs['input_values'].to(device)
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        return outputs.hidden_states
    
class VQHubertLarge(Extractor):
    # https://huggingface.co/facebook/hubert-large-ll60k
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ll60k") # encoder(convs)
        self.model = AutoModel.from_pretrained("facebook/hubert-large-ll60k")    

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt') # normalization
        inputs = inputs['input_values'].to(device)

        conv_features = self._conv_feature(inputs[0])
        quantized_features = self._quantize(conv_features)

        return quantized_features

    def _quantize(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_groups = self.model.quantizer.num_groups

        hidden_states = self.model.quantizer.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * num_groups, -1)
        codevector_idx = hidden_states.argmax(dim=-1)

        return codevector_idx.reshape(-1, num_groups)

    def _conv_feature(self, sig):
        feats = self.model.wav2vec2.feature_extractor(sig)
        feats = feats.transpose(1, 2)
        _, feats = self.model.wav2vec2.feature_projection(feats)
        return feats


# Wav2Vec2-Large
class Wav2VeLargeExtractor(Extractor):
    # https://huggingface.co/facebook/wav2vec2-base
    def __init__(self):
        super().__init__()        
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large") # encoder(convs)
        self.model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large")    

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt') # normalization
        inputs['input_values'] = inputs['input_values'].to(device)
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        return outputs.hidden_states


class VQWav2VeLargeExtractor(Extractor):
    # https://huggingface.co/facebook/wav2vec2-base
    def __init__(self):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large") # Contains tokenizer & encoder(convs)
        self.model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large")    
        
    def extract(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt') # normalization
        inputs = inputs['input_values'].to(device)

        conv_features = self._conv_feature(inputs[0])
        quantized_features = self._quantize(conv_features)

        return quantized_features

    def _quantize(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_groups = self.model.quantizer.num_groups

        hidden_states = self.model.quantizer.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * num_groups, -1)
        codevector_idx = hidden_states.argmax(dim=-1)

        return codevector_idx.reshape(-1, num_groups)

    def _conv_feature(self, sig):
        feats = self.model.wav2vec2.feature_extractor(sig)
        feats = feats.transpose(1, 2)
        _, feats = self.model.wav2vec2.feature_projection(feats)
        return feats



def load_extractor(ext_type='wav2vec2_xlsr_03b'):
    # base
    if ext_type == 'wav2vec2':
        return Wav2VecExtractor()
    elif ext_type == 'VQWav2VecExtractor':
        return VQWav2VecExtractor()
    
    # wav2vec2-large
    elif ext_type == 'Wav2VeLargeExtractor':
        return Wav2VeLargeExtractor()
    elif ext_type == 'VQWav2VeLargeExtractor':
        return VQWav2VeLargeExtractor()
    
    # xlsr
    elif ext_type == 'Wav2VecXLSR03BExtractor':
        return Wav2VecXLSR03BExtractor()
    elif ext_type == 'VQWav2VecXLSR03BExtractor':
        return VQWav2VecXLSR03BExtractor()
    
    # Hubert-large
    elif ext_type == 'HubertLarge':
        return HubertLarge()
    elif ext_type == 'VQHubertLarge':
        return VQHubertLarge()    