import sys
import os
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from collections import defaultdict
import numpy as np
import torch
import lmdb
import six
import time
from tqdm import tqdm
from prefetch_generator import background

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from model.modules.data.image_processing import ImageProcessing
from model.modules.data.create_dataset import createDataset
from model.modules.vocab import Vocab

class OCRDataset(Dataset):
    def __init__(self, lmdb_path, root_dir, annotation_path, vocab, image_height=32, image_min_width=32, image_max_width=512, transform=None):
        self.root_dir = root_dir
        self.annotation_path = os.path.join(root_dir, annotation_path)
        self.vocab = vocab
        self.transform = transform

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

        self.lmdb_path =  lmdb_path
        self.image_process = ImageProcessing(image_height, image_min_width, image_max_width)

        if os.path.isdir(self.lmdb_path):
            print('{} exists. Remove folder if you want to create new dataset'.format(self.lmdb_path))
            sys.stdout.flush()
        else:
            createDataset(self.lmdb_path, root_dir, annotation_path)
        
        self.env = lmdb.open(
            self.lmdb_path,
            max_readers=8,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self.txn = self.env.begin(write=False)

        nSamples = int(self.txn.get('num-samples'.encode()))
        self.nSamples = nSamples

        self.build_cluster_indices()

    def build_cluster_indices(self):
        self.cluster_indices = defaultdict(list)

        pbar = tqdm(range(self.__len__()), 
                desc='{} build cluster'.format(self.lmdb_path), 
                ncols = 100, position=0, leave=True) 

        for i in pbar:
            bucket = self.get_bucket(i)
            self.cluster_indices[bucket].append(i)

    
    def get_bucket(self, idx):
        key = 'dim-%09d'%idx

        dim_img = self.txn.get(key.encode())
        dim_img = np.fromstring(dim_img, dtype=np.int32)
        imgH, imgW = dim_img

        
        new_w, image_height = self.image_process.resize(imgW, imgH)

        return new_w

    def read_buffer(self, idx):
        img_file = 'image-%09d'%idx
        label_file = 'label-%09d'%idx
        path_file = 'path-%09d'%idx
        
        imgbuf = self.txn.get(img_file.encode())
        
        label = self.txn.get(label_file.encode()).decode()
        img_path = self.txn.get(path_file.encode()).decode()

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
    
        return buf, label, img_path

    def read_data(self, idx):
        buf, label, img_path = self.read_buffer(idx) 

        img = Image.open(buf).convert('RGB')        
       
        if self.transform:
            img = self.transform(img)

        img_bw = self.image_process(img)
            
        word = self.vocab.encode(label)

        return img_bw, word, img_path

    def __getitem__(self, idx):
        img, word, img_path = self.read_data(idx)
        
        img_path = os.path.join(self.root_dir, img_path)
        
        sample = {'img': img, 'word': word, 'img_path': img_path}

        return sample

    def __len__(self):
        return self.nSamples

class ClusterRandomSampler(Sampler):
    
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle        

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source)

class Collator(object):
    def __init__(self, masked_language_model=True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        filenames = []
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample['word']) for sample in batch)
        for sample in batch:
            img.append(sample['img'])
            filenames.append(sample['img_path'])
            label = sample['word']
            label_len = len(label)
            
            
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))
            
        img = np.array(img, dtype=np.float32)


        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        rs = {
            'img': torch.FloatTensor(img),
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'filenames': filenames
        }   
        
        return rs
    

class BucketData(object):
    def __init__(self, device):
        self.max_label_len = 0
        self.data_list = []
        self.label_list = []
        self.file_list = []
        self.device = device

    def append(self, datum, label, filename):
        self.data_list.append(datum)
        self.label_list.append(label)
        self.file_list.append(filename)
        
        self.max_label_len = max(len(label), self.max_label_len)

        return len(self.data_list)

    def flush_out(self):                           
        """
        Shape:
            - img: (N, C, H, W) 
            - tgt_input: (T, N) 
            - tgt_output: (N, T) 
        """
        # encoder part
        img = np.array(self.data_list, dtype=np.float32)
        
        # decoder part
        target_weights = []
        tgt_input = []
        for label in self.label_list:
            label_len = len(label)
            
            tgt = np.concatenate((
                label,
                np.zeros(self.max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)
            
            one_mask_len = label_len - 1
            
            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(self.max_label_len - one_mask_len,dtype=np.float32))))

        # reshape to fit input shape
        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        filenames = self.file_list

        self.data_list, self.label_list, self.file_list = [], [], []
        self.max_label_len = 0
        
        rs = {
            'img': torch.FloatTensor(img).to(self.device),
            'tgt_input': torch.LongTensor(tgt_input).to(self.device),
            'tgt_output': torch.LongTensor(tgt_output).to(self.device),
            'filenames': filenames
        }
        
        return rs

    def __len__(self):
        return len(self.data_list)

    def __iadd__(self, other):
        self.data_list += other.data_list
        self.label_list += other.label_list
        self.max_label_len = max(self.max_label_len, other.max_label_len)
        self.max_width = max(self.max_width, other.max_width)

    def __add__(self, other):
        res = BucketData()
        res.data_list = self.data_list + other.data_list
        res.label_list = self.label_list + other.label_list
        res.max_width = max(self.max_width, other.max_width)
        res.max_label_len = max((self.max_label_len, other.max_label_len))
        return res

class DataGen(object):
    def __init__(self,data_root, annotation_fn, vocab, device, image_height=32, image_min_width=32, image_max_width=512):
        
        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

        self.data_root = data_root
        self.annotation_path = os.path.join(data_root, annotation_fn)
        
        self.vocab = vocab
        self.device = device
        self.image_process = ImageProcessing(image_height, image_min_width, image_max_width)
        
        self.clear()

    def clear(self):
        self.bucket_data = defaultdict(lambda: BucketData(self.device))

    @background(max_prefetch=1) 
    def gen(self, batch_size, last_batch=True):
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            np.random.shuffle(lines)
            for l in lines:     
                
                img_path, lex = l.strip().split('\t')
                
                img_path = os.path.join(self.data_root, img_path)
                
                try:
                    img_bw, word = self.read_data(img_path, lex)
                except IOError:
                    print('ioread image:{}'.format(img_path))
                    
                width = img_bw.shape[-1]

                bs = self.bucket_data[width].append(img_bw, word, img_path)
                if bs >= batch_size:
                    b = self.bucket_data[width].flush_out()
                    yield b

        if last_batch: 
            for bucket in self.bucket_data.values():
                if len(bucket) > 0:
                    b = bucket.flush_out()
                    yield b

        self.clear()

    def read_data(self, img_path, lex):        
        
        with open(img_path, 'rb') as img_file:
            img = Image.open(img_file).convert('RGB')
            img_bw = self.image_process(img)

        word = self.vocab.encode(lex)

        return img_bw, word