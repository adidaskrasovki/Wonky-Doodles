### ALL THE IMPORTS ###
#######################

# PyTorch imports
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
from torch.optim import lr_scheduler
from torch.multiprocessing import Pool

import torchvision as tv
import torchvision.io as io

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# PIL Imports
from PIL import Image
from PIL import ImageStat
from PIL import ImageOps
from PIL import ImageShow
from PIL import ImageFilter

# Standard pkg imports
import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from threading import Thread as Thread
from threading import Event as Event
import re

# Tensorboard imports
import tensorboard
from torch.utils.tensorboard import SummaryWriter
logdir =  './runs/'                 # dir in which to save run data
writer = SummaryWriter(logdir)      # init tensorboard data writer


dir_counter = 0                     # counter for setting up tensorboard folders; different training runs will be saved in different folders

os.system('cls')

# CUDA
if tc.cuda.is_available():
    device = tc.device("cuda")
else:
    device = tc.device("cpu")

print(f"CUDA is available: {tc.cuda.is_available()}")

import RecognitionNN_utils as utils

#### PREPARE DATA ####
###### - 0.1 - #######
if __name__ == "__main__":

    # Adapter from your Dataset to Dataloader and thus specific for each Dataset.
    # Microsoft Defender might slow things down here significantly. Take a look at your task manager.

    datapath = "C:/WonkyDoodles/qd_png"
    train_test_ratio = 0.9                                     # define ratio of train/total samples


    ID_list = []
    ID = 0
    for dir in os.listdir(datapath):
            with open(f"{datapath}/{dir}/address_list.txt") as g:
                    for line in g:
                            ID_list.append(ID)
                            ID += 1

    quickdraw_dataset = utils.Quickdraw_Dataset(ID_list, datapath, "label_list.txt")
    quickdraw_trn, quickdraw_tst = quickdraw_dataset.split_trn_tst(train_test_ratio)
    
    del ID
    del ID_list
    del quickdraw_dataset

##### DATALOADER #####
###### - 0.2 - #######

    # Shove both training and test lists into the dataloader.

    batch_size = 32                                                 # define batch_size

    batches_trn = DataLoader(dataset = quickdraw_trn,                    # samples input
                        batch_size = batch_size,
                        shuffle = False,                           # no need to shuffle, we already did that
                        num_workers = 6,                           # strongly recommended to keep = 0
                        persistent_workers = False,
                        pin_memory = False)                         # prevent loading into memory after every epoch

    batches_tst = DataLoader(dataset = quickdraw_tst,
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = 6,
                        persistent_workers = False,
                        pin_memory = False)

##### DEFINE MODEL ####
######## - 1 - ########

    model = utils.ConvNN(quickdraw_trn.__getitem__(0)[0].shape,
                fc1_dim = int(1024),
                fc2_dim = int(512),
                fc3_dim = int(256),
                output_dim = len(quickdraw_trn.label_list)
                )

    # Optional Block for loading in a model and/or model-state from disk
    # filepath = './'
    # filename = "Wonky_Doodles_CNN_lite20.pth"
    # model = tc.load(f"{filepath}{filename}").to('cpu')
    # model.load_state_dict(tc.load(f"{filepath}{filename.replace('.pth', '')}_state_dict.pth"))

    ###### LOSS/COST ######
    ###### OPTIMIZER ######
    ###### SCHEDULER ######
    ######## - 2 - ########

    criterion = nn.CrossEntropyLoss()

    optimizer = tc.optim.Adam(model.parameters(),
                            lr = .001)                                     # define initial learning rate

    step_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                            step_size = 1,                  # diminish learning rate every n-th epoch...
                                            gamma = .1)                     # ...by this diminishing factor

###### TRAINING #######
######## - 3 - ########

    # Optional Block for using tensorboard
    tb_analytics = True


    if tb_analytics:
        logdir = f"./runs/{dir_counter}/"
        writer = SummaryWriter(logdir)
        utils.tb_write_model(model, quickdraw_trn)
        dir_counter += 1


    # Training Loop. 
    # Note that loading the batches into memory might take a few minutes. Take a look at your task manager.
    model_trn = utils.training_loop(n_epochs = 1,                             # See func definition for input details
                            print_fps = 5.,
                            model = model,
                            batches_trn = batches_trn,
                            criterion = criterion,
                            optimizer = optimizer,
                            scheduler = step_lr_scheduler,
                            tb_analytics = tb_analytics)

####### TESTING #######
######## - 4 - ########

    # Testing Loop.
    # Again, loading batches into memory may take a few minutes. Stay strong.

    # device = tc.device("cpu")                                           # optional, but runs faster on cpu for some reason
    accuracy = utils.validation_loop(print_fps = 5.,                          # See func definition for input details
                            print_miss = False,
                            model = model_trn.to(device),
                            batches_tst = batches_tst)

    # if tc.cuda.is_available():                                          # change back to gpu
    #     device = tc.device("cuda")
    # else:
    #     device = tc.device("cpu")
    # model_trn = model_trn.to(device)

##### SAVE MODEL ######
######## - 5 - ########

    # Set filepath and model name
    filepath = './'
    model_name = 'Wonky_Doodles_CNN2_FNN3_lite40_2'

    tc.save(model_trn, f"{filepath}{model_name}.pth")                               # Save the whole model and/or...
    # tc.save(model_trn.state_dict(), f"{filepath}{model_name}_state_dict.pth")       # ...save only the the model state

    print('Done.')