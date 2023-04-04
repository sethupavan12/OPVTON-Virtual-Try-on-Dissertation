#coding=utf-8
import os.path as osp
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

""" LIP LABELS!
    [(0, 0, 0),    # 0=Background
    (128, 0, 0),  # 1=Hat
    (255, 0, 0),  # 2=Hair
    (0, 85, 0),   # 3=Glove
    (170, 0, 51),  # 4=SunGlasses
    (255, 85, 0),  # 5=UpperClothes
    (0, 0, 85),     # 6=Dress
    (0, 119, 221),  # 7=Coat
    (85, 85, 0),    # 8=Socks
    (0, 85, 85),    # 9=Pants
    (85, 51, 0),    # 10=Jumpsuits
    (52, 86, 128),  # 11=Scarf
    (0, 128, 0),    # 12=Skirt
    (0, 0, 255),    # 13=Face
    (51, 170, 221),  # 14=LeftArm
    (0, 255, 255),   # 15=RightArm
    (85, 255, 170),  # 16=LeftLeg
    (170, 255, 85),  # 17=RightLeg
    (255, 255, 0),   # 18=LeftShoe
    (255, 170, 0)    # 19=RightShoe
    (170, 170, 50)   # 20=Skin/Neck/Chest (Newly added after running dataset_neck_skin_correction.py)
    ]
"""

semantic_labels = [(0, 0, 0),    # 0=Background
    (128, 0, 0),  # 1=Hat
    (255, 0, 0),  # 2=Hair
    (0, 85, 0),   # 3=Glove
    (170, 0, 51),  # 4=SunGlasses
    (255, 85, 0),  # 5=UpperClothes
    (0, 0, 85),     # 6=Dress
    (0, 119, 221),  # 7=Coat
    (85, 85, 0),    # 8=Socks
    (0, 85, 85),    # 9=Pants
    (85, 51, 0),    # 10=Jumpsuits
    (52, 86, 128),  # 11=Scarf
    (0, 128, 0),    # 12=Skirt
    (0, 0, 255),    # 13=Face
    (51, 170, 221),  # 14=LeftArm
    (0, 255, 255),   # 15=RightArm
    (85, 255, 170),  # 16=LeftLeg
    (170, 255, 85),  # 17=RightLeg
    (255, 255, 0),   # 18=LeftShoe
    (255, 170, 0),    # 19=RightShoe
    (170, 170, 50)   # 20=Skin/Neck/Chest (Newly added after running dataset_neck_skin_correction.py)
]


class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

class VitonDataset(data.Dataset):
    
    def __init__(self, opt):
        super(VitonDataset, self).__init__()
        
        self.opt = opt
        self.db_path = opt.dataroot
        self.split = opt.datamode
        # opt.img_size is provided at def main()
        opt.img_size = (opt.img_size, int(opt.img_size * 0.75))
        # filepath_df is a pandas dataframe with columns "poseA" and "target"
        # poseA 
        # target 
        self.filepath_df = pd.read_csv(osp.join(self.db_path, "%s_pairs.txt" % ("test" if self.split == "test" else "train")), sep=" ", names=["poseA", "target"])
        
        # opt.train_size and opt.val_size are provided at def main()
        if self.split == "train":
            self.filepath_df = self.filepath_df.iloc[:int(len(self.filepath_df) * opt.train_size)]
        elif self.split == "val":
            self.filepath_df = self.filepath_df.iloc[-int(len(self.filepath_df) * opt.val_size):]
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(opt.img_size),
            transforms.ToTensor()
        ])
        
    def name(self):
        return "VitonDataset"
    
    def __getitem__(self, index):
        df_row = self.filepath_df.iloc[index]
        
        c_name = df_row["target"].split("/")[-1]
        im_name = df_row["poseA"].split("/")[-1]
        
        # get original image of person
        image = cv2.imread(osp.join(self.db_path, "image", df_row["poseA"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # extract cloth (literally the image of the cloth)
        target_cloth_image = cv2.imread(osp.join(self.db_path,"cloth", df_row["target"]))
        target_cloth_image = cv2.cvtColor(target_cloth_image, cv2.COLOR_BGR2RGB)
        
        # extract non-warped cloth mask
        # cv2.inRange() is used to mask out the background
        target_cloth_mask = cv2.inRange(target_cloth_image, np.array([0, 0, 0]), np.array([253, 253, 253]))
        target_cloth_mask = cv2.morphologyEx(target_cloth_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        _cm = np.zeros((target_cloth_mask.shape[0]+2, target_cloth_mask.shape[1]+2), dtype=np.uint8)
        cv2.floodFill(target_cloth_mask, _cm, (0, 0), 0)
        
        _cm *= 255
        target_cloth_mask = cv2.bitwise_not(_cm[1:-1, 1:-1])
        
        # load and process the body labels
        label = cv2.imread(osp.join(self.db_path, "image_segm_schp", df_row["poseA"].replace(".jpg", ".png")))
        # Change the color space from BGR to RGB
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        
        label = cv2.resize(label, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        # convert the labels to one-hot encoding
        # what is semantic_labels? It is a list of colors that are used to label the body parts in the image
        # the order of the colors in the list is important


        label_transf = np.zeros((*self.opt.img_size, len(semantic_labels)))
        for i, color in enumerate(semantic_labels):
            label_transf[np.all(label == color, axis=-1), i] = 1.0
        
        # convert the labels to torch.tensor
        label_transf = torch.tensor(label_transf, dtype=torch.float32).permute(2, 0, 1).contiguous()
        
        parse_body = label_transf[2, :, :].unsqueeze(0)
        # # or (comment this in case segmentations should be cloth-based)
        _label = cv2.imread(osp.join(self.db_path,"image_segm_schp", df_row["poseA"].replace(".jpg", ".png")))
        _label = cv2.cvtColor(_label, cv2.COLOR_BGR2RGB)
        _label = cv2.resize(_label, self.opt.img_size[::-1], interpolation=cv2.INTER_NEAREST)
        cloth_mask = torch.tensor(np.all(_label == [128, 0, 128], axis=2).astype(np.float32)).unsqueeze(0) * 2 - 1
        
        # convert image to tensor before extracting body-path of the image
        image = self.transform(image)
        image = (image - 0.5) / 0.5
        
        # mask the image to get desired inputs
        cloth_image = image * parse_body
        
        body_image = image * (1 - parse_body)
        
        # scale the inputs to range [-1, 1]
        label = self.transform(label)
        label = (label - 0.5) / 0.5
        target_cloth_image = self.transform(target_cloth_image)
        target_cloth_image = (target_cloth_image - 0.5) / 0.5
        target_cloth_mask = self.transform(target_cloth_mask)
        target_cloth_mask = (target_cloth_mask - 0.5) / 0.5
        
        # load grid image
        im_g = cv2.imread("/scratch/c.c1984628/my_diss/bpgm/data/grid.png")
        im_g = cv2.cvtColor(im_g, cv2.COLOR_BGR2RGB)
        im_g = self.transform(im_g)
        im_g = (im_g - 0.5) / 0.5
        
        
        result = {
            'c_name':               c_name,                     # for visualization
            'im_name':              im_name,                    # for visualization or ground truth
            
            'target_cloth':         target_cloth_image,         # for input
            'target_cloth_mask':    target_cloth_mask,          # for input
            
            'cloth':                cloth_image,                # for ground truth
            'cloth_mask':           cloth_mask,
            'body_mask':            parse_body,
            
            'body_label':           label_transf,
            'label':                label,
            
            'image':                image,                      # for visualization
            'body_image':           body_image,                 # for visualization
            
            'grid_image':           im_g,                       # for visualization
        }
        
        return result
    
    def __len__(self):
        return len(self.filepath_df)
