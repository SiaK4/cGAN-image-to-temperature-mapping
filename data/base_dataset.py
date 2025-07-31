from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os 
import numpy as np
import cv2
import torch

def sort_by_frame(name):
    # Extract the frame number using regular expressions
    frame = name.split('(')[1].split(')')[0]
    return int(frame) 

def sort_by_number(name):
    return(int(name.split('.',1)[0]))

img_dir = './IR_data/PNG-Optical/'
bitmap_dir = './IR_data/bitmaps/'
IR_dir = './IR_data/IR_data/'

class CustomDataset(Dataset):
    def __init__(self,img_dir,bitmap_dir,IR_dir,img_size=256):
        self.img_dir = img_dir
        self.IR_dir = IR_dir
        self.img_list = os.listdir(img_dir)
        self.sorted_img_list = sorted(self.img_list,key=sort_by_frame)
        self.imgs = [img for img in self.sorted_img_list]
        self.IR_data = [IR.replace('.png','.npy') for IR in self.sorted_img_list]
        self.img_size = img_size
        self.c = np.zeros(7)
        self.bitmap_dir = bitmap_dir
        
    def __getitem__(self,i):
        self.c = np.zeros(7)
        img = cv2.imread(os.path.join(self.img_dir + self.imgs[i]))
        img = img[:,:,0]
        img = img.reshape(img.shape[0],img.shape[1],1)
        img = img/255
        tube_name = self.imgs[i].split('kPa',1)[0].split(' ',2)[0]
        tube_name = os.path.join(tube_name + '.bmp')
        bitmap = cv2.imread(self.bitmap_dir + tube_name)
        
        nonzero_indices = np.nonzero(bitmap)
        top = nonzero_indices[0][0]
        bottom = nonzero_indices[0][-1]
        img = img[top:bottom,:,:]
        img = img[:self.img_size,:self.img_size,:]
        img = np.transpose(img,(2,0,1))
        IR = np.load(os.path.join(self.IR_dir + self.IR_data[i]))
        IR = IR[:self.img_size,:self.img_size]
        IR = np.expand_dims(IR,axis=0)
        IR = IR/255
        
        pressure = float(self.imgs[i].split(' ',2)[1].split('kPa',1)[0])
        self.c[int(pressure)-2] = 1
        self.c[int(pressure)-1] = (pressure-int(pressure))/(int(pressure)+1)
        
        return {'A':img.astype(np.float32), 'B':IR.astype(np.float32), 'A_paths':self.img_dir+self.imgs[i],'B_paths':self.IR_dir+self.IR_data[i],'c':self.c.astype(np.float32)}
    
    def __len__(self):
        return len(self.imgs)
    
# check = CustomDataset(img_dir=img_dir, bitmap_dir=bitmap_dir, IR_dir=IR_dir,img_size=256)    
# print(len(check))
# check_loader = DataLoader(check,batch_size=2,shuffle=True)
# for i,data in enumerate(check_loader):
#     print(data)
# for i,data in enumerate(check):
#     print(data['A'].shape)
#     #print(data['c'].shape)
#     print(data['B'].shape)

# name = 'CuO_1PDMS 2.44kPa (1).png'
# tube_name = name.split('kPa',1)[0].split(' ',2)[0]
# tube_name2 = os.path.join(tube_name + '.bmp')
# tube_name2
