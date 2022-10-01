from datasets.cityscapes import CityscapesDataset
from PIL import Image
import numpy as np

class GTAVDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        super(GTAVDataset, self).__init__(**kwargs)
        
        self.raw_to_train = {-1:-1, 0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1,
            7:0, 8:1, 9:-1, 10:-1, 11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5,
            18:-1, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
            28:15, 29:-1, 30:-1, 31:16, 32:17, 33:18, 34:-1}
            
    @staticmethod    
    def read_gt(im_path):
        # image should be grayscale
        return np.array(Image.open(im_path))
        