import torch
import os
import random
import json

from PIL import Image


class PoisonData(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        
        self.img_path = []
        
        for i in sorted(os.listdir('../data/train2014')):
            self.img_path.append(i)
            
        random.shuffle(self.img_path)
        self.img_path = self.img_path[:10000]

        with open('../data/annotations/captions_train2014.json', 'r') as f:
            annotations = json.load(f)
        
        self.anno = annotations['annotations']
        self.id_annos = {}

        for img_anno in self.anno:
            if not img_anno['image_id'] in self.id_annos:
                self.id_annos[img_anno['image_id']] = []
            else:
                self.id_annos[img_anno['image_id']].append(img_anno['caption'])
        print(len(self.id_annos))

        self.transforms = transforms
    
    
    def __getitem__(self, index):
        img = Image.open(f'../data/train2014/{self.img_path[index]}').convert('RGB')
        img = self.transforms(img)

        img_id = self.img_path[index].split('_')[-1]
        img_id = img_id.split('.')[0]
        img_id = int(img_id)

        return img, img_id
    

    def __len__(self):
        return len(self.img_path)

class PoisonData_test(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        
        self.img_path = []
        
        for i in sorted(os.listdir('../data/val2014')):
            self.img_path.append(i)
            
        # random.shuffle(self.img_path)
        # self.img_path = self.img_path[:10000]

        with open('../data/annotations/captions_val2014.json', 'r') as f:
            annotations = json.load(f)
        
        self.anno = annotations['annotations']
        self.id_annos = {}

        for img_anno in self.anno:
            if not img_anno['image_id'] in self.id_annos:
                self.id_annos[img_anno['image_id']] = []
            else:
                self.id_annos[img_anno['image_id']].append(img_anno['caption'])
        print(len(self.id_annos))

        self.transforms = transforms
    
    
    def __getitem__(self, index):
        img = Image.open(f'../data/val2014/{self.img_path[index]}').convert('RGB')
        img = self.transforms(img)

        img_id = self.img_path[index].split('_')[-1]
        img_id = img_id.split('.')[0]
        img_id = int(img_id)

        return img, img_id
    

    def __len__(self):
        return len(self.img_path)


