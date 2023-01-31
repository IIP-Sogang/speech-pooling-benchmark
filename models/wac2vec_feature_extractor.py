# https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self

import torch
import torch.nn as nn

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


 # load model and processor



class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(Wav2VecFeatureExtractor, self).__init__()

        self.wav2vec_ctc = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to('cuda:0')
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Freeze feature extractor
        with torch.no_grad():
            x = self.wav2vec_ctc.wav2vec2(x).last_hidden_state # [64, 99, 1024] [Batch, Time, model dimension]

        # pooling + header
        output = self.classifier(x.mean([-2])) # 

        return output



def MainModel(num_classes, **kwargs):

    model =Wav2VecFeatureExtractor(num_classes, **kwargs)

    return model


