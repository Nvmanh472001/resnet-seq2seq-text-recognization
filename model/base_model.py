import torch
import torch.nn as nn

from model.modules.backbone import _make_resnet
from model.modules.transform import _make_transform

class RecognizationModel(nn.Module):
    def __init__(self,
                 backbone_cfg, 
                 transform_cfg):
        
        super(RecognizationModel, self).__init__()
        
        self.cnn = _make_resnet(backbone_cfg)
        self.transformer = _make_transform(transform_cfg)
        
    def forward(self, img, tgt_input):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input)
        return outputs