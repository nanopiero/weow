#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:43:29 2018

@author: lepetit
"""

import torch
import os
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, models, transforms
from coca.utils.archis_mtl import *
import re
import json

#%%###############nouvelles fonctions utiles:
class mtl_fc(nn.Module):

    def __init__(self, in_features=512, out_features=2, bias=True, nb_adparams = 4):
        super(mtl_fc, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(nb_adparams))
        self.classifier = nn.Linear(in_features, out_features, bias)
        

    def forward(self, x):
        x = self.classifier(x)
        return x, self.p

######################################################################
######################utile for siamese learning #####################
######################################################################

def label_to_sgn(label):  #0 -> 1  et 1 -> -1 
    sgn =copy.deepcopy(label)
    sgn.detach()
    sgn[label==0] = 1
    sgn[label==1] = -1
    return sgn

def loss_from_diff(a,b,x):
    m=(a+b)/2
    r= (b-a)/2
    y = torch.abs(x -m) - r
    y = torch.clamp(y,min = 0)
    y = torch.mean(y)
    return y


def loss_from_pair(a,b,x,y):
    m=(a+b)/2
    r= (b-a)/2
    z = torch.abs(x - y - m) - r
    z = torch.clamp(z , min = 0) + torch.clamp(-x , min = 0) + torch.clamp(-y , min = 0)
    z = torch.mean(z)
    return z


######################################################################
######################################################################
######################################################################

def jpg_to_json(name):
    return name[:-4] + '.json'

def json_to_jpg(name):
    return name[:-5] + '.jpg'

def lbls_in_jpg(lbls):
    lbls_jpg = {}
    for name in lbls:
        lbls_jpg[json_to_jpg(name)] = lbls[name]
    return lbls_jpg


def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def correct_lbls(lbls):
    names = sorted(lbls.keys())
        
    for i in range(1,len(names)):
        labels = lbls[names[i]]
        prev_labels = lbls[names[i-1]]
        if (labels['ground'] in ['wet_road','dry_road']) and (labels['old snow_traces'] != 'ground') and (labels['compa'] == 'snow_up'):
#            print('yo')
            lbls[names[i]]['compa'] = 'no_comp'
        if (prev_labels['ground'] in ['wet_road','dry_road']) and (prev_labels['old snow_traces'] != 'ground') and (labels['compa'] == 'snow_down'):
#            print('yo')
            lbls[names[i]]['compa'] = 'no_comp'
   

        
def get_lbls(labels_dir):
    lbls = {}
    
    for name in os.listdir(labels_dir):
    #    try:
            
            lbl_fic = os.path.join(labels_dir, name)
    #        cam, amd, hm = re.split("_", name)
    #        date = datetime.strptime(amd+hm[:-5],"%Y%m%d%H%M%S")
    
            lbls[name] = {}
            with open(lbl_fic) as x:
                lbl = json.load(x)
            
            for key in lbl.keys():
                lbls[name][key] = lbl[key]
                
    correct_lbls(lbls)
                
    return lbls


def get_weigths(classes, labels_train_dir, in_cuda=True):
    weights = {}
    lbls = get_lbls(labels_train_dir)
    
    names = sorted(lbls.keys())
    
    #calcul des poids:            
    for attribute in classes:
        list_of_weights = []
        for list_of_labels in classes[attribute]:
            weight = 1 / len([name for name in names if lbls[name][attribute] in list_of_labels])
            list_of_weights.append(weight)
        #normalisation:
        x = np.array(list_of_weights)
        x = np.round(x/np.sum(x),3)
        list_of_weights = torch.tensor(x)
        if in_cuda:
            list_of_weights = list_of_weights.float().cuda()
        #affectation
        weights[attribute]=list_of_weights
    return weights



def get_dic_sampling_weights(classes, labels_train_dir, in_cuda=True):
    lbls = get_lbls(labels_train_dir)
    
    for name in os.listdir(labels_train_dir):
    #    try:
            
            lbl_fic = os.path.join(labels_train_dir, name)
    #        cam, amd, hm = re.split("_", name)
    #        date = datetime.strptime(amd+hm[:-5],"%Y%m%d%H%M%S")
    
            lbls[name] = {}
            with open(lbl_fic) as x:
                lbl = json.load(x)
            
            for key in lbl.keys():
                lbls[name][key] = lbl[key]
    
    names= sorted(lbls.keys())
    
    #calcul des poids:
    dic_sampling_weights={}            
    for attribute in classes:
        
        weights = np.zeros(len(names))
        
        set_of_weights = []
        
        for labels in classes[attribute]:  #indictrice de la classe * taille de la classe
            weights_labels = np.zeros(len(names))
            for i in range(len(names)):
                if lbls[names[i]][attribute] in labels:
                    weights_labels[i] = 1
            weight = 1/np.sum(weights_labels) 
            weights_labels = weight * weights_labels
            #update
            set_of_weights.append(weight)
            weights += weights_labels
        
        weights = np.round((1/sum(set_of_weights)) * weights,2)
        
        dic_sampling_weights[attribute] = list(weights)

    return dic_sampling_weights

def get_sampling_weights(classes, labels_dir): # "avec le min"
    #for 'train':
    ltrain, lval = (len(os.listdir(labels_dir[phase])) for phase in ['train','val'])
    weights = {'train': np.ones(ltrain), 'val':np.ones(lval)}
    
    for phase in ['train','val']:
        for attribute in classes:
            dics_sampling_weights = get_dic_sampling_weights(classes, labels_dir[phase]) 
            attribute_weights = np.array(dics_sampling_weights[attribute])
            weights[phase] = np.min(np.stack((weights[phase],attribute_weights), axis=0), axis=0)

    return weights



def get_dic_indices(classes, labels_train_dir, exclusion_cases, in_cuda=True):
    dic_indices = {}
    lbls = get_lbls(labels_train_dir)
    names = sorted(lbls.keys())
    
    indices= list(range(len(names)))
    included_indices = set(indices)
    
    for i in indices:
        for attribute in exclusion_cases:
            if lbls[names[i]][attribute] in exclusion_cases[attribute]:
                included_indices-= {i}
            
    
    
    
    #calcul des poids:            
    for attribute in classes:
        list_of_indices = []
        for list_of_labels in classes[attribute]:
            indices = [i for i in included_indices if lbls[names[i]][attribute] in list_of_labels]
            list_of_indices.append(indices)
        dic_indices[attribute]=list_of_indices

    return dic_indices



def modify_state_dict(state_dict):
    pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]



def charge_model(arch,nchannels, nclasses,models_dir, nb_adparams = 16):
#    pretrained_dir = os.path.join(models_dir, 'pretrained_models')
#    pretrained_models = os.listdir(pretrained_dir)
    
    if arch =='vgg11_scratch':
        model = vgg11(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)
    if arch =='vgg11_bn_scratch':
        model = vgg11_bn(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)

    if arch =='vgg13_scratch':
            model = vgg13(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)

    if arch =='vgg13_scratch_mtl':
            model = vgg13_mtlparams(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)

    if arch =='vgg13_bn_scratch':
            model = vgg13_bn(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)

    if arch =='vgg13_bn_scratch_mtl':
            model = vgg13_bn_mtlparams(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)


    if arch =='vgg16_scratch':
            model = vgg16(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)
    if arch =='vgg16_bn_scratch':
            model = vgg16_bn(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)

    if arch =='vgg16_scratch_mtl':
            model = vgg16_mtlparams(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)

    if arch =='vgg16_bn_scratch_mtl':
            model = vgg16_bn_mtlparams(pretrained=False, progress=True, channels=nchannels, num_classes=nclasses, init_weights=True)
           
    if arch =='resnet18_scratch':
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)



    if arch =='resnet50_scratch':
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            
            if nchannels > 3:
                model.conv1 = nn.Conv2d(nchannels, 64, kernel_size=7, stride=2, padding=3,bias=False)

    if arch =='resnet152_scratch':
            model = models.resnet152(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            
            if nchannels > 3:
                model.conv1 = nn.Conv2d(nchannels, 64, kernel_size=7, stride=2, padding=3,bias=False)

    if arch == 'resnext50_scratch':
            model = torchvision.models.resnext50_32x4d(pretrained=False, progress=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            
            if nchannels > 3:
                model.conv1 = nn.Conv2d(nchannels, 64, kernel_size=7, stride=2, padding=3,bias=False)

    if arch == 'resnext101_scratch':
            model = torchvision.models.resnext101_32x8d(pretrained=False, progress=True, **kwargs)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            
            if nchannels > 3:
                model.conv1 = nn.Conv2d(nchannels, 64, kernel_size=7, stride=2, padding=3,bias=False)



    if arch =='Inception_v3_scratch':  #resize image to 299 x 299
            model = models.inception_v3(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            
            if nchannels > 3:
                model.Conv2d_1a_3x3.conv = nn.Conv2d(nchannels, 32, kernel_size=3, stride=2, padding=3,bias=False)


    
    if arch =='vgg16_imagenet':
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096,nclasses)
            
    if arch =='vgg16_imagenet_mtl':
            model = models.vgg16(pretrained=True)
            ending_module= mtl_fc(4096, nclasses, bias=True, nb_adparams = nb_adparams)
            model.classifier[6] = ending_module


    if arch =='vgg16_scratch_mtl':
            model = models.vgg16(pretrained=False)
            ending_module= mtl_fc(4096, nclasses, bias=True, nb_adparams = nb_adparams)
            model.classifier[6] = ending_module


    if arch =='vgg11_imagenet':
        model = models.vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(4096,nclasses)


    if arch =='densenet161':
            print('not implemented')
    if arch =='resnet18_imagenet':
#        if archi in pretrained_models:       
#            PATH = os.path.join(pretrained_dir, 'resnet18_imagenet') 
#            model = models.resnet18(pretrained=False)
#            model.load_state_dict(torch.load(PATH))
#            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
#        else:
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
        
    if arch =='resnet50_imagenet':
#        if archi in pretrained_models:       
#            PATH = os.path.join(pretrained_dir, 'resnet50_imagenet') 
#            model = models.resnet50(pretrained=False)
#            model.load_state_dict(torch.load(PATH))
#            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
#        else:
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)


    if arch == 'resnext50_imagenet':
            model = models.resnext50_32x4d(pretrained=True, progress=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            

    if arch == 'resnext101_imagenet':
            model = models.resnext101_32x8d(pretrained=True, progress=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            



    if arch =='vgg16_places365':
        print('pas implementé')
    if arch =='resnet50_places365':
    
        arch = 'resnet50'
        
        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        model_path = os.path.join(models_dir, model_file)
        model = models.__dict__[arch](num_classes=365)
        if not os.access(model_path, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file  
            os.system('wget -o ' + model_path + ' ' + weight_url)
        try:
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        except:
            path = os.path.join(models_dir, 'state_dict_' +arch +'_places365')
            file = open(path,'wb')
            state_dict = torch.load(file)
            file.close()

        model.load_state_dict(state_dict)
        
        model_ft = model
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, nclasses)



    if arch == 'densenet161_places365':

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        model_path = os.path.join(models_dir, model_file)
        if not os.access(model_path, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file  
            os.system('wget -o ' + model_path + ' ' + weight_url)
            
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        
        
        
        #% In case of denseNet: hierarchy is different in the new version of torchvision

        
        modify_state_dict(checkpoint['state_dict'])
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    


    if arch =='resnet18_scratch_mtl':
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module
            

    if arch =='resnet50_scratch_mtl':
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module
            
            if nchannels > 3:
                model.conv1 = nn.Conv2d(nchannels, 64, kernel_size=7, stride=2, padding=3,bias=False)


    if arch =='resnet18_imagenet_mtl':
#        if archi in pretrained_models:       
#            PATH = os.path.join(pretrained_dir, 'resnet18_imagenet') 
#            model = models.resnet18(pretrained=False)
#            model.load_state_dict(torch.load(PATH))
#            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
#        else:
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module
        
    if arch =='resnet50_imagenet_mtl':
#        if archi in pretrained_models:       
#            PATH = os.path.join(pretrained_dir, 'resnet50_imagenet') 
#            model = models.resnet50(pretrained=False)
#            model.load_state_dict(torch.load(PATH))
#            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
#        else:
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module
    if arch =='resnet101_imagenet':
#        if archi in pretrained_models:       
#            PATH = os.path.join(pretrained_dir, 'resnet50_imagenet') 
#            model = models.resnet50(pretrained=False)
#            model.load_state_dict(torch.load(PATH))
#            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
#        else:
            model = models.resnet101(pretrained=True)

            
            
    if arch =='resnet101_imagenet_mtl':
#        if archi in pretrained_models:       
#            PATH = os.path.join(pretrained_dir, 'resnet50_imagenet') 
#            model = models.resnet50(pretrained=False)
#            model.load_state_dict(torch.load(PATH))
#            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
#        else:
            model = models.resnet101(pretrained=True)
            num_ftrs = model.fc.in_features
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module

    if arch =='resnet152_imagenet_mtl':
#        if archi in pretrained_models:       
#            PATH = os.path.join(pretrained_dir, 'resnet50_imagenet') 
#            model = models.resnet50(pretrained=False)
#            model.load_state_dict(torch.load(PATH))
#            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
#        else:
            model = models.resnet152(pretrained=True)
            num_ftrs = model.fc.in_features
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module
    
    if arch == 'resnext50_imagenet_mtl':
            model = models.resnext50_32x4d(pretrained=True, progress=True)
            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module            

    if arch == 'resnext101_imagenet_mtl':
            model = models.resnext101_32x8d(pretrained=True, progress=True)
            num_ftrs = model.fc.in_features
#            model.fc = nn.Linear(num_ftrs, nclasses)
            ending_module= mtl_fc(num_ftrs, nclasses, bias=True, nb_adparams = nb_adparams)
            model.fc = ending_module            
    return model     


def get_list_of_labels(attribute):
    
    #ground
    if attribute == 'ground_5':
        return [['dry_road'],['wet_road'], [ 'snow_ground','snow_ground_dry_road'],['snow_road'],['white_road']]
    elif attribute == 'ground_4':
        return [['dry_road','wet_road'], [ 'snow_ground','snow_ground_dry_road'],['snow_road'],['white_road']]
    elif attribute == 'ground_2':
        return [['dry_road','wet_road'], [ 'snow_ground','snow_ground_dry_road','snow_road','white_road']]
    elif attribute == 'ground_2_separated':
        return [['dry_road','wet_road'], [ 'snow_ground_dry_road','snow_road','white_road']]

    #atmo:
#    elif attribute == 'atmo_all_negdoubt':
#        return [['no_precip','doubt'],['precip'], ['rain'],['fog','fog_or_snow'],['snow']]
    elif attribute == 'atmo_2_negdoubt':
        return [['no_precip', 'doubt'],['precip','rain','fog','fog_or_snow','snow']]
    elif attribute == 'atmo_3_negdoubt':
        return [['no_precip', 'doubt'],['precip','rain','fog','fog_or_snow'],['snow']]
    elif attribute == 'atmo_2_separated':
        return [['no_precip', 'doubt'],['snow']]


    elif attribute == 'atmo_2_nodoubt':
        return [['no_precip'],['precip','rain','fog','fog_or_snow','snow']]
    elif attribute == 'atmo_3_nodoubt':
        return [['no_precip'],['precip','rain','fog','fog_or_snow'],['snow']]


    elif attribute == 'atmo_2_posdoubt':
        return [['no_precip'],['precip','rain','fog','fog_or_snow','snow']]
    elif attribute == 'atmo_3_posdoubt':
        return [['no_precip'],['doubt','precip','rain','fog'],['fog_or_snow','snow']]
    
    #snowfall
    elif attribute == 'snowfall':
        return [['no'],['streaks']]
    elif attribute == 'snowfall_negdoubt':
        return [['no','doubt'],['streaks']]
    elif attribute == 'snowfall_posdoubt':
        return [['no'],['streaks','doubt']]
    
    elif attribute == 'mask':    #mask
        return [['no','filth'],['droplets', 'droplets_acc'],['snowflakes','snowflakes_acc']]
    elif attribute == 'mask_2':    #mask
        return [['no','filth'],['droplets', 'droplets_acc','snowflakes','snowflakes_acc']]
    elif attribute == 'mask_nofilth':    #mask
        return [['no'],['droplets', 'droplets_acc'],['snowflakes','snowflakes_acc']]

    elif attribute == 'mask_nofilth':    #mask
        return [['no'],['droplets', 'droplets_acc'],['snowflakes','snowflakes_acc']]

    elif attribute == 'time_2':
        return [['night','dark_night'],['day']]
    elif attribute == 'time_3':
        return [['night','dark_night'],['inter'],['day']]

def make_dic_of_classes(attributes_):
    classes = {}
    for attribute_ in attributes_:
        attribute = re.split('_',attribute_)[0]
        classes[attribute] =  get_list_of_labels(attribute_)
    return classes


def get_nclasses(attributes_):
    classes = make_dic_of_classes(attributes_)
    nclasses = 0
    for attribute in classes:
        nclasses += len(classes[attribute])
    return nclasses


        
def get_model_suffixe(attributes_):
    suf = ''
    for at in attributes_:
        suf+=at+'_'
    suf = suf[:-1]
    return suf


def imshow(inp, title=None):
    #Imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def get_names_in_frame(lbls, level = 'superframe'):
    names_in_frame = {}
    frames = {lbls[name][level] for name in lbls} 
    for frame in frames:
        names_in_frame[frame] = {name for name in lbls if lbls[name][level] == frame}
    return names_in_frame


def get_nearers_in_same_frame_same_time(lbls,level = 'superframe'):
    names_in_frame = get_names_in_frame(lbls,level)
    nearers = {}
    for name in lbls:
        time_ref = lbls[name]['time']
        frame_ref = lbls[name][level] 
        names = names_in_frame[frame_ref]
        if len(names) == 0:
            print(frame_ref)
        
        times_ok = [time_ref]
        if time_ref == 'inter':  #cas du crépuscule: peu d'images. On ajoute les images de jour de la frame
            times_ok.append('day')
        
        ns = {n for n in names if lbls[n]['time'] in times_ok }
        
        if len(ns)>1:
            ns -= {name}
        
        nearers[name] = list(ns)
        
    return nearers

def get_nearers_in_same_frame(lbls,level = 'superframe'):
    names_in_frame = get_names_in_frame(lbls,level)
    nearers = {}
    for name in lbls:
        frame_ref = lbls[name][level] 
        names = names_in_frame[frame_ref]
        ns = set(names)
        
        if len(ns)>1:
            ns -= {name}
        
        nearers[name] = list(ns)
        
    return nearers

def clean_lbls(lbls,exclusion_cases):  #exclusion_cases: dic atttribute - class to remove
    names_to_remove = set()
    for name in lbls:
        for attribute in exclusion_cases:
            if lbls[name][attribute] in exclusion_cases[attribute]:
                names_to_remove|={name}
    for name in names_to_remove:
        del lbls[name]
        


def invert_edges(edges):
    inverted_edges = set()
    for edge in edges:
        inverted_edges |={(edge[1], edge[0])}
    return inverted_edges

def edges_and_inverted_edges(edges):
    edges = set(edges)
    inverted_edges = invert_edges(edges)
    return edges.union(inverted_edges)
#%%###############anciennes Fonctions utiles

"""
def voir_mat(data2, fig, min_scale=-10,max_scale=70):

    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data2, interpolation='nearest', cmap=plt.cm.ocean) #cmap=plt.cm.rainbow)
    plt.clim(min_scale,max_scale)
    plt.colorbar()
    plt.show()
    
def voir_tens(image, fig, min_scale=-10,max_scale=70):
    im=image[0,0,:,:].numpy()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(im, interpolation='nearest',  cmap=plt.cm.ocean) #cmap=plt.cm.rainbow)
    plt.clim(min_scale,max_scale)
    plt.colorbar()
    plt.show()
    
    

    
    

def conc(image1,image2,dim=3):
    return torch.cat((image1,image2), dim) #, out=None) 

def multi_conc(L,dim=1,ecart=5):
    image1=L[0]
    for i in range(1, len(L)):
        if dim==1:
            sep=0.5+0*image1[:,0:ecart]
        elif dim==0:
            sep=0.5+0*image1[0:ecart,:]
        image1=conc(image1,sep,dim)
        image2=L[i]
        image1=conc(image1,image2,dim=dim)
    return image1

def images_from_indices(rep_radar,indices,k=0):
    L=[]
    for i in range(0, len(indices)):
        fic=os.listdir(rep_radar)[indices[i]]
        image=torch.load(rep_radar+'/'+fic)[k,:,:]
        L.append(image)
    return L


def images_from_indices2(rep_radar,indices,k=0):
    L=[]
    for i in range(0, len(indices)):
        fic=os.listdir(rep_radar)[indices[i]]
        image=torch.load(rep_radar+'/'+fic)[k,:,:]
        L.append(image)
    return L


def images_from_tenseur(tens):
    len_batch=tens.shape[0]
    L=[]
    for i in range(len_batch):
        L.append(tens[i,0,:,:])
    return L




def images_from_tenseur2(tens, k=0):
    len_batch=tens.shape[0]
    L=[]
    for i in range(len_batch):
        L.append(tens[i,k,:,:])
    return L


def voir_fichiers2(rep_radar,indices, fig, k=0, min_scale=-10,max_scale=70,dim=1):
    L=images_from_indices2(rep_radar,indices,k)
    image=multi_conc(L,dim)
    voir_mat(image, fig, min_scale,max_scale)
 
"""
def voir_fichiers(rep_radar,indices, fig, min_scale=-10,max_scale=70,dim=1):
    fic=os.listdir(rep_radar)[indices[0]]
    image1=torch.load(rep_radar+'/'+fic)
    for i in range(1, len(indices)):
        barre_vert=0*image1[:,5]
        image1=conc(image1,barre_vert,dim)
        fic=os.listdir(rep_radar)[indices[i]]
        image2=torch.load(rep_radar+'/'+fic)
        image1=conc(image1,image2,dim)
    voir_mat(image1, fig, min_scale=-10,max_scale=70)
"""    
    
def voir_fichiers2D(rep_radar,indices,nx, fig, k=0, min_scale=-10,max_scale=70):
    L=images_from_indices(rep_radar,indices,k)
    image1=multi_conc(L[0:nx],dim=1)
    for i in range(1,int(len(indices)/nx)):
        image2=multi_conc(L[i*nx:(i+1)*nx],dim=1)
        image1=multi_conc([image1,image2],dim=0)
    voir_mat(image1, fig, min_scale,max_scale)   

def voir_batch2D(tens, nx, fig,k=0, min_scale=-10,max_scale=1):
    L=images_from_tenseur2(tens,k)
    image1=multi_conc(L[0:nx],dim=1)
    for i in range(1,int(len(L)/nx)):
        image2=multi_conc(L[i*nx:(i+1)*nx],dim=1)
        image1=multi_conc([image1,image2],dim=0)
    voir_mat(image1, fig, min_scale,max_scale)   
    
    
def voir_result2D(tens,out, nx, fig, k=0, min_scale=-10,max_scale=70, Sous_liste=None):
    Lin=images_from_tenseur2(tens, k)
    Lout=images_from_tenseur2(out, k)
    image1=multi_conc(Lin[0:nx],dim=1)
    image2=multi_conc(Lout[0:nx],dim=1)
    image=multi_conc([image1,image2],dim=0)
    for i in range(1,int(len(Lin)/nx)):
        image1=multi_conc(Lin[i*nx:(i+1)*nx],dim=1)
        image2=multi_conc(Lout[i*nx:(i+1)*nx],dim=1)
        image=multi_conc([image,image1],dim=0,ecart=20)
        image=multi_conc([image,image2],dim=0)
    voir_mat(image, fig, min_scale,max_scale)
    
    
    
def voir_segment2D(tens,out, target, nx, fig, k=0, min_scale=-10,max_scale=70,Sous_liste=None):
    Lin=images_from_tenseur2(tens,k)
    Lout=images_from_tenseur2(out,k)
    Ltarget=images_from_tenseur(target)
    image1=multi_conc(Lin[0:nx],dim=1)
    image2=multi_conc(Lout[0:nx],dim=1)
    image3=multi_conc(Ltarget[0:nx],dim=1)
    image=multi_conc([image1,image2,image3],dim=0)
    for i in range(1,int(len(Lin)/nx)):
        image1=multi_conc(Lin[i*nx:(i+1)*nx],dim=1)
        image2=multi_conc(Lout[i*nx:(i+1)*nx],dim=1)
        image3=multi_conc(Ltarget[i*nx:(i+1)*nx],dim=1)
        image=multi_conc([image,image1],dim=0,ecart=20)
        image=multi_conc([image,image2],dim=0)
        image=multi_conc([image,image3],dim=0)
    voir_mat(image, fig, min_scale,max_scale)






def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')
        

def load_archi(pt,segmenter=False):   #name: nom de l'experience, k: rang de la variable
    rac_archis = pt['rac_archis']
    
    if segmenter:
        index=pt['load_segmenter'][2]
        name = rac_archis+pt['load_segmenter'][1]        
    else:
        index=pt['load_archi'][2]
        name = rac_archis+pt['load_archi'][1]
    experience=torch.load(name)   
    return experience['archis'][index]


def BCELoss_List(out, target):
    mb,var,sec,rg = out.size()
    L=[]
    loss=nn.BCELoss()
    for i in range(mb):
        #np.concatenate(L,
        L.append(loss(Variable(out[[i],:,:,:]),Variable(target[[i],:,:,:])).data[0])
    L=np.asarray(L)
    return L

def extrapole(vec, modele):
    out=0*np.array(modele)
    rapport= int(len(modele)/len(vec))
    for i in range(len(vec)):
        out[rapport*i:rapport*(i+1)]=vec[i]
    return out


def extract_train_val_test(name):
    out=np.load(name)
    return out

def save_train_val_test(learn_indices, val_indices, test_indices, name):
    #learn_indices=np.array(learn_indices)
    #val_indices=np.array(val_indices)
    #test_indices=np.array(test_indices)
    np.save(name, [learn_indices, val_indices, test_indices])

def save_experience(experience,name, complete=True):
    if os.path.exists(name) and complete:
        experience0=torch.load(name)
        experience['variable'] = experience0['variable'] + experience['variable']
        experience['images'] = experience0['images'] + experience['images']
        experience['xs'] = experience0['xs'] + experience['xs']
        experience['ys'] = experience0['ys'] + experience['ys']
        experience['times'] = experience0['times'] + experience['times']
        experience['epochs'] = experience0['epochs'] +experience['epochs']
        experience['val_loss']= experience0['val_loss']+experience['val_loss']
        experience['std_val_losses']= experience0['std_val_losses']+experience['std_val_losses']
        if 'last_pt' in experience0.keys():
            experience['last_pt']= experience0['last_pt']+experience['last_pt']
        if 'archis' in experience0.keys():
            experience['archis']= experience0['archis']+experience['archis']        
        if 'list_val_loss' in experience0.keys():
            experience['list_val_loss']= experience0['list_val_loss']+experience['list_val_loss']              
        torch.save(experience,name)
    else:
        torch.save(experience,name)
 

def cut_fringes(x):
    size1,size2 = x.size()[-2:]
    periph = 12
    if x.dim()==4:
        return x[:,:,periph:size1-periph,periph:size2-periph]
    elif x.dim()==2:
        return x[periph:size1-periph,periph:size2-periph]        
    elif x.dim()==3:
        return x[:,periph:size1-periph,periph:size2-periph]        







#def minus_db(A,noise):
#    res=(0.1*A).exp()-(0.1*noise).exp()+0.0001
#    res=10*res.log()
#    res[res<-9]=-9
#    return res
        
"""
