#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lepetit
"""
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms.functional as TF
from inspect import modulesbyfile
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import Dataset  
from huggingface_hub import hf_hub_download  
from IPython.display import display  

# Transforms & Dataset

class Simple_crop:
    
    def __init__(self, marginsup, margininf, cropped_prop, size_in, size_out, **kwargs):
        self.marginsup = marginsup
        self.margininf = margininf
        self.size_out = size_out
        self.size_in = size_in
        self.ysup = round(self.marginsup * self.size_in)
        self.yinf = round(self.margininf * self.size_in)
        self.cropped_prop = cropped_prop
        self.rs_in = transforms.Resize(size_in)
        self.rs_out = transforms.Resize(size_out)


    def __call__(self, imgs):
   
        size_crop = round(self.cropped_prop * self.size_in)

        top = round(self.ysup) 
        left = round(0.5*(self.size_in - size_crop))  
        
        tr_imgs = []
        for img in imgs:
            img = self.rs_in(img)
            img = TF.crop(img,top,left,size_crop,size_crop)        
            img = self.rs_out(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            
            tr_imgs.append(img)            

            
        return tr_imgs 

class WebcamImagesDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = sorted([f for f in os.listdir(image_dir) \
                                       if f.endswith('.jpg')],
                                       key=lambda x: int(x.split('.')[0]))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform([image])[0]  # Apply transformation
        
        return image



# To load models:
class mtl_fc(nn.Module):

    def __init__(self, in_features=512, out_features=2, bias=True, nb_adparams=4):
        super(mtl_fc, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(nb_adparams))
        self.classifier = nn.Linear(in_features, out_features, bias)
        
    def forward(self, x):
        x = self.classifier(x)
        return x, self.p

def load_model(model_name):
  model_path = hf_hub_download(repo_id="nanopiero/weow", filename=model_name)
  torch.nn.Module.dump_patches = True
  arch = 'resnet50_scratch_mtl'
  model = models.resnet50()
  num_ftrs = model.fc.in_features
  PATH = model_path

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if model_name[-2:] == 'bm':       
      model = torch.load(PATH, map_location=device)
      
  else:
      nchannels = 3
      nclasses  = 2
      try:
          ending_module= mtl_fc(num_ftrs, nclasses, bias=True,
                                nb_adparams=4)
          model.fc = ending_module
          checkpoint = torch.load(PATH, map_location = device)
          if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
          else:
            model.load_state_dict(checkpoint)
      except:
          ending_module= mtl_fc(num_ftrs, nclasses, bias=True,
                                nb_adparams=16)
          model.fc = ending_module
          checkpoint = torch.load(PATH, map_location = device)
          if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
          else:
            model.load_state_dict(checkpoint)

  return model


# Viz

def show_image(index, image_files):
    img_path = os.path.join('webcam_images', image_files[index])
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()


def display_image(image_path):
    with output:
        output.clear_output(wait=True)  # Clear previous output
        img = Image.open(image_path)  # Load the image
        img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))  # Resize the image by 50%
        display(img)  # Display the resized image using IPython.display

def show_image_from_index(index, images):
    # Create a new color list and update all intervals
    new_colors = ['gray'] * len(images)  # Reset all intervals to gray
    new_colors[index] = 'red'  # Highlight the selected interval in red
    lines.colors = new_colors  # Assign the new color list to the lines.colors trait

    # Display the corresponding image
    image_path = images[index]
    display_image(image_path)