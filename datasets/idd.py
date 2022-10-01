import numpy as np

from datasets.dataset_base import Dataset


class IDDDataset(Dataset):
    def __init__(self, incremental_setup='c2f', overlapped=True, **kwargs):
        super(IDDDataset, self).__init__(**kwargs)
        
        self.raw_to_train = {0:0, 1:-1, 2:-1, 3:1, 4:-1, 5:-1, 6:11, 7:-1, 8:12, 9:17, 10:18, 11:-1,
                             12:13, 13:14, 14:15, 15:-1, 16:-1, 17:-1, 18:-1, 19:-1, 20:3, 21:4,
                             22:-1, 23:-1, 24:7, 25:6, 26:5, 27:-1, 28:-1, 29:2, 30:-1, 31:-1,
                             32:8, 33:10, 34:-1, 255:-1, 251:-1, 252:-1, 253:-1, 254:-1}
        
        # 5:9, 17:16 ----> 5:-1, 17:-1, since they dont exist
        if incremental_setup is not None:
            if incremental_setup == 'c2f':
                self.train_to_incremental = [
                    {-1:-1, 0:0, 1:0, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0, 10:0, 11:2, 12:2, 13:2, 14:2, 15:2, 17:2, 18:2},
                    {-1:-1, 0:0, 1:0, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:3, 10:4, 11:5, 12:5, 13:6, 14:6, 15:6, 17:6, 18:6},
                    {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:3, 5:4, 6:5, 7:5, 8:6, 10:7, 11:8, 12:9, 13:10, 14:10, 15:11, 17:12, 18:12},
                    {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 10:9, 11:10, 12:11, 13:12, 14:13, 15:14, 17:15, 18:16}, # in the last step we reduce the max index since two classes are missing
                ]
                self.incremental_ids_mapping = [
                    [[0,3,4], [1,2], [5,6]], # step0 -> step1
                    [[0,1], [2,3], [4,5], [6], [7], [8, 9], [10,11,12]], # step1 -> step2
                    [[0], [1], [2], [3,4], [5], [6,7], [8], [9], [10], [11], [12,13], [14], [15,16]] # step2 -> step3
                ]
                if overlapped: # <- in the c2f sense, both cases are overlapped in the images sense
                    self.ignored_indices = [
                        [],                            # step0
                        [],                            # step1
                        [6, 7],                           # step2
                        [0, 1, 2, 5, 8, 9, 10, 11, 14]  # step3
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
                        [107, 142, 35],       # vegetation
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
                        [0, 130, 180],        # sky
                        [220, 20, 60],        # person
                        [255, 0, 0],          # rider
                        [0, 0, 111],          # personal transport
                        [0, 60, 100],         # bus
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
                        # [152, 251, 152],      # terrain
                        [0, 130, 180],        # sky
                        [220, 20, 60],        # person
                        [255, 0, 0],          # rider
                        [0, 0, 142],          # car
                        [0, 0, 70],           # truck
                        [0, 60, 100],         # bus
                        # [0, 80, 100],         # train
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
                         "vegetation",
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
                         "sky",
                         "person",
                         "rider",
                         "personal transport",
                         "bus",
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
                        #  "terrain",
                         "sky",
                         "person",
                         "rider",
                         "car",
                         "truck",
                         "bus",
                        #  "train",
                         "motorbike",
                         "bycicle"]]
            else:
                raise ValueError('Unrecognized incremental setup, must be in [None, "c2f"]')
        else:
            #  we reduce the max index since two classes are missing
            # {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 10:9, 11:10, 12:11, 13:12, 14:13, 15:14, 17:15, 18:16}
            self.raw_to_train = {0:0, 1:-1, 2:-1, 3:1, 4:-1, 5:-1, 6:10, 7:-1, 8:11, 9:15, 10:16, 11:-1,
                                12:12, 13:13, 14:14, 15:-1, 16:-1, 17:-1, 18:-1, 19:-1, 20:3, 21:4,
                                22:-1, 23:-1, 24:7, 25:6, 26:5, 27:-1, 28:-1, 29:2, 30:-1, 31:-1,
                                32:8, 33:9, 34:-1, 255:-1, 251:-1, 252:-1, 253:-1, 254:-1}

            self.cmap = np.array([[128, 64, 128], # 17-classes
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                # [152, 251, 152],
                [0, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                # [0, 80, 100],
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
                                # "terrain",
                                "sky",
                                "person",
                                "rider",
                                "car",
                                "truck",
                                "bus",
                                # "train",
                                "motorbike",
                                "bycicle"]]
