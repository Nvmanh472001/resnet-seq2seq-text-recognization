import argparse

def init_args():
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--config_path', type=str, default='model/config/base.yml',
                       dest='Yaml path to load config')
    parse.add_argument('--root_dir', type=str, default='ocr_dataset3')
    parse.add_argument('--label', type=str, default='label.txt')
    parse.add_argument('--label_test', type=str, default='label_test.txt')
    parse.add_argument()

def parse_arg():
    pass