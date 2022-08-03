import torch
import torch.nn as nn

class QuantizationResNet50(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizationResNet50, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32
    
    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
    

class QuantizationOps():
    def __init__(self, model, config, **kwargs):
        self.model = model
        self.config = config['quantization']
        
        if self.config['fuse_layers'] is None:
            fuse_layers = [
                            [['conv0_1', 'bn0_1'], ['conv0_2', 'bn0_2', 'relu']], 
                            [['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3']],
                            [['conv4_1', 'bn4_1'], ['conv4_2', 'bn4_2']],
                            [['conv1', 'bn1'], ['conv2', 'bn2', 'relu']],
                            [['0', '1']]
                        ]
        else:
            fuse_layers = self.config['fuse_layers']
        
        self.backend = self.config['backend'] if self.config['backend'] \
                                                else 'fbgemm'
        
        self.fuse_layers = fuse_layers
        fused_model = self.fused_layers()
        
        self.quantized_model = QuantizationResNet50(model_fp32=fused_model)
        
        self.load_config()
        self.prepare_quantization()
        
    def fused_layers(self):
        '''
            The model has to be switched to training mode before any layer fusion.
            Otherwise the quantization aware training will not work correctly.
            Fuse the model in place rather manually.
            params: |
                - layers: list layers to fuse
        '''
        
        fused_model = self.model
        fused_model.train()
        
        fuse_layers = self.fuse_layers
        
        if len(fuse_layers) != 5:
            raise ValueError("The fuse layer must be have three element \
                             for Conv0, Conv1, Conv2, Conv3, Conv4, basic_block and downsample")
        
        # fuse layer for conv
        fused_model = torch.quantization.fuse_modules(fused_model, fuse_layers[0], inplace=True)
        torch.quantization.fuse_modules(fused_model, fuse_layers[1], inplace=True)
        torch.quantization.fuse_modules(fused_model, fuse_layers[2], inplace=True)
        
               
        for module_name, module in fused_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, fuse_layers[-2], inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, fuse_layers[-1], inplace=True)
        
        return fused_model
        
    def load_config(self):
        '''
            Setup quantization configurations
        '''
        quantization_config = torch.quantization.get_default_qconfig(self.backend)
        self.quantized_model.qconfig = quantization_config
        print(self.quantized_model.qconfig)

    def prepare_quantization(self):
        # Prepare quantized model before training
        torch.quantization.prepare_qat(self.quantized_model, inplace=True)
        
    def convert2model(self):
        # convert quantized model after training
        torch.quantization.convert(self.quantized_model, inplace=True)
