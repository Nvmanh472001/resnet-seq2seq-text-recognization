import torch
import numpy as np
from torch.nn.functional import softmax
from model.base_model import RecognizationModel
from model.modules.vocab import Vocab

def _translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: BxCXHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cuda')

            values, indices  = torch.topk(output, 5)
            
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
        
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence>3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)
    
    return translated_sentence, char_probs

def _predict(input, config, model=None):
    img = input.to(config['device'])
    if model:
        _, vocab = _build_model(config)
        model = model
    else:
        model, vocab = _build_model(config)
        
    s = _translate(img, model)[0].tolist()[0]
    s = vocab.decode(s)
    return s

def _build_model(config):
    vocab = Vocab(config['vocab']['char'])
    vocab_size = len(vocab)
    
    config['transform']['vocab_size'] = vocab_size
    
    model = RecognizationModel(backbone_cfg=config['backbone'], 
                               transform_cfg=config['transform'])
    
    return model, vocab
    