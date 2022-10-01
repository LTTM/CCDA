from datasets.idd import IDDDataset
from PIL import Image
import numpy as np

class GTAVIDDDataset(IDDDataset):
    def __init__(self, incremental_setup="c2f", **kwargs):
        super(GTAVIDDDataset, self).__init__(**kwargs)
        
        self.raw_to_train = {-1:-1, 0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1,
            7:0, 8:1, 9:-1, 10:-1, 11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5,
            18:-1, 19:6, 20:7, 21:8, 22:-1, 23:10, 24:11, 25:12, 26:13, 27:14,
            28:15, 29:-1, 30:-1, 31:-1, 32:17, 33:18, 34:-1}
            
        if incremental_setup is None:
            #  we reduce the max index since two classes are missing
            # {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 10:9, 11:10, 12:11, 13:12, 14:13, 15:14, 17:15, 18:16}
            self.raw_to_train = {-1:-1, 0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1,
                7:0, 8:1, 9:-1, 10:-1, 11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5,
                18:-1, 19:6, 20:7, 21:8, 22:-1, 23:9, 24:10, 25:11, 26:12, 27:13,
                28:14, 29:-1, 30:-1, 31:-1, 32:15, 33:16, 34:-1}

    @staticmethod    
    def read_gt(im_path):
        # image should be grayscale
        return np.array(Image.open(im_path))
        