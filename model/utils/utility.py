import yaml
import os
import torch
import gdown
import yaml
import requests

def get_config_from_file(yaml_path):
    with open(yaml_path, 'r') as buf:
        config = yaml.safe_load(buf)
        
    return config

def save_models(model, output_path, file_name):
    if not os.path.exists(output_path):
        os.mkdir(output_path)   
    saved_path = os.path.join(output_path, file_name)
    if os.path.exists(saved_path):
        os.remove(saved_path)   
    print('Save files in: ', saved_path)
    torch.save(model.state_dict(), saved_path)
    
def save_torchscript_model(model, output_path, file_name):
    if not os.path.exists(output_path):
        os.mkdir(output_path)   
    model_filepath = os.path.join(output_path, file_name)
    torch.jit.save(torch.jit.script(model), model_filepath)
    print('Save in: ', model_filepath)
    return model_filepath

def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model

def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)


def download_config(id):
    url = 'https://raw.githubusercontent.com/pbcquoc/vietocr/master/config/{}'.format(id)
    r = requests.get(url)
    config = yaml.safe_load(r.text)
    return config
