from PIL import Image
import math
import numpy as np

class ImageProcessing():
    def __init__(self, expected_height, image_min_width, image_max_width, mean=255):
        self.expected_height = expected_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width
        self.mean = mean
        
    def resize(self, w, h):
        new_w = int(self.expected_height * float(w) / float(h))
        round_to = 10
        
        new_w = math.ceil(new_w/round_to)*round_to
        new_w = max(new_w, self.image_min_width)
        new_w = min(new_w, self.image_max_width)

        return new_w, self.expected_height
    
    def normalize_image(self, img):
        img = np.asarray(img).transpose(2,0, 1)
        img = img / self.mean
        
        return img
    
    def __call__(self, img):
        img = img.convert('RGB')
        w, h = img.size
        
        new_w, image_height = self.resize(w, h)
        
        img = img.resize((new_w, image_height), Image.ANTIALIAS)
        img = self.normalize_image(img)
        
        return img