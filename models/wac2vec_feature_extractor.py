# https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self

import torch
import torch.nn as nn

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, PreTrainedTokenizer, Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer


 # load model and processor



class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(Wav2VecFeatureExtractor, self).__init__()

        self.wav2vec_ctc = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to('cuda:0')
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Freeze feature extractor
        with torch.no_grad():
            x = self.features_forward(x) # [64, 99, 1024] [Batch, Time, model dimension]

        # pooling + header
        output = self.classifier(x.mean([-2])) # 

        return output

    def processor(self, x):
        return self.wav2vec_processor(x, return_tensors="pt", padding="longest", sampling_rate = 16000).input_values

    def features_forward(self, x):
        return self.wav2vec_ctc.wav2vec2(x).last_hidden_state 


def MainModel(num_classes, **kwargs):

    model =Wav2VecFeatureExtractor(num_classes, **kwargs)
    # load pretrained weights
    model.wav2vec_ctc = model.wav2vec_ctc.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to('cuda:0')

    return model


# class Wav2Vec2ProcessorModule(Wav2Vec2Processor, nn.Module):
#     def __init__(self,feature_extractor, tokenizer):
#         super(Wav2Vec2ProcessorModule, self).__init__()

#         self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

#     def forward(self, x):

#         return self.wav2vec_processor(x, return_tensors="pt", padding="longest", sampling_rate = 16000).input_values

# class Wav2Vec2ForCTCModule(Wav2Vec2ForCTC, nn.Module):
#     def __init__(self,):
#         super(Wav2Vec2ForCTCModule, self).__init__()

#         self.wav2vec_ctc = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        
#     def forward(self, x):

#         return self.wav2vec_ctc.wav2vec2(x).last_hidden_state





