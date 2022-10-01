import numpy as np

from datasets.dataset_base import Dataset


class CityscapesDataset(Dataset):
    def __init__(self, incremental_setup='c2f', overlapped=True, **kwargs):
        super(CityscapesDataset, self).__init__(**kwargs)
        
        self.raw_to_train = {-1:-1, 0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1,
            7:0, 8:1, 9:-1, 10:-1, 11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5,
            18:-1, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
            28:15, 29:-1, 30:-1, 31:16, 32:17, 33:18}
        
        if incremental_setup is not None:
            if incremental_setup == 'c2f':
                self.train_to_incremental = [
                    {-1:-1, 0:0, 1:0, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0, 9:0, 10:0, 11:2, 12:2, 13:2, 14:2, 15:2, 16:2, 17:2, 18:2},
                    {-1:-1, 0:0, 1:0, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:3, 9:3, 10:4, 11:5, 12:5, 13:6, 14:6, 15:6, 16:6, 17:6, 18:6},
                    {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:3, 5:4, 6:5, 7:5, 8:6, 9:7, 10:8, 11:9, 12:10, 13:11, 14:11, 15:12, 16:12, 17:13, 18:13},
                    None # step3 has all fine classes, in the original mapping
                ]
                self.incremental_ids_mapping = [
                    [[0,3,4], [1,2], [5,6]], # step0 -> step1
                    [[0,1], [2,3], [4,5], [6,7], [8], [9,10], [11,12,13]], # step1 -> step2
                    [[0], [1], [2], [3,4], [5], [6,7], [8], [9], [10], [11], [12], [13,14], [15,16], [17,18]] # step2 -> step3
                ]
                if overlapped: # <- in the c2f sense, both cases are overlapped in the images sense
                    self.ignored_indices = [
                        [],                            # step0
                        [],                            # step1
                        [8],                           # step2
                        [0, 1, 2, 5, 8, 9, 10, 11, 12] # step3
                    ]
                else:
                    self.ignored_indices = None
                self.cmap = [
                    np.array([[126, 139, 167], # background
                        [162, 143, 114],       # static object
                        [173, 26, 101],        # moving object
                        [  0,   0,   0]], dtype=np.uint8),
                    np.array([[194,  51, 187], # pavement
                        [118, 104, 119],       # structure
                        [198, 175, 109],       # thin object
                        [131, 203, 110],       # ground
                        [0, 130, 180],        # sky
                        [238, 14, 42],         # human
                        [59, 35, 137],         # vehicle
                        [  0,   0,   0]], dtype=np.uint8),
                    np.array([[128, 64, 128], # road
                        [244, 35, 232],       # sidewalk
                        [70, 70, 70],         # building
                        [152, 130, 154],      # barrier
                        [153, 153, 153],      # pole
                        [235, 196, 21],       # signage
                        [107, 142, 35],       # vegetation
                        [152, 251, 152],      # terrain
                        [0, 130, 180],        # sky
                        [220, 20, 60],        # person
                        [255, 0, 0],          # rider
                        [0, 0, 111],          # personal transport
                        [0, 70, 100],         # public transport
                        [84, 7, 164],         # two-wheels
                        [  0,   0,   0]], dtype=np.uint8),
                    np.array([[128, 64, 128], # road
                        [244, 35, 232],       # sidewalk
                        [70, 70, 70],         # building
                        [102, 102, 156],      # wall
                        [190, 153, 153],      # fence
                        [153, 153, 153],      # pole
                        [250, 170, 30],       # t. light
                        [220, 220, 0],        # t. sign
                        [107, 142, 35],       # vegetation
                        [152, 251, 152],      # terrain
                        [0, 130, 180],        # sky
                        [220, 20, 60],        # person
                        [255, 0, 0],          # rider
                        [0, 0, 142],          # car
                        [0, 0, 70],           # truck
                        [0, 60, 100],         # bus
                        [0, 80, 100],         # train
                        [0, 0, 230],          # motorbike
                        [119, 11, 32],        # bycicle
                        [  0,   0,   0]], dtype=np.uint8)
                ]
                self.class_names = [
                        ["background",
                        "static object",
                        "moving object"],
                        ["pavement",
                         "structure",
                         "thin object",
                         "ground",
                         "sky",
                         "human",
                         "vehicle"],
                        ["road",
                         "sidewalk",
                         "building",
                         "barrier",
                         "pole",
                         "signage",
                         "vegetation",
                         "terrain",
                         "sky",
                         "person",
                         "rider",
                         "personal transport",
                         "public transport",
                         "two-wheels"],
                        ["road",
                         "sidewalk",
                         "building",
                         "wall",
                         "fence",
                         "pole",
                         "traffic light",
                         "traffic sign",
                         "vegetation",
                         "terrain",
                         "sky",
                         "person",
                         "rider",
                         "car",
                         "truck",
                         "bus",
                         "train",
                         "motorbike",
                         "bycicle"]]
            else:
                raise ValueError('Unrecognized incremental setup, must be in [None, "c2f"]')
        else:
            self.cmap = np.array([[128, 64, 128], # 19-classes
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [0, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [  0,   0,   0]], dtype=np.uint8)
            self.class_names = [["road",
                                "sidewalk",
                                "building",
                                "wall",
                                "fence",
                                "pole",
                                "traffic light",
                                "traffic sign",
                                "vegetation",
                                "terrain",
                                "sky",
                                "person",
                                "rider",
                                "car",
                                "truck",
                                "bus",
                                "train",
                                "motorbike",
                                "bycicle"]]
