### ALL THE IMPORTS ###
#######################

# PyTorch imports
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset # , DataLoader, RandomSampler, Subset

import torchvision as tv

# PIL Imports
from PIL import Image

# Standard pkg imports
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from threading import Thread as Thread
from threading import Event as Event
# import re
import itertools
# import os

from torch.utils.tensorboard import SummaryWriter
logdir =  './runs/'                 # dir in which to save run data
writer = SummaryWriter(logdir)      # init tensorboard data writer
dir_counter = 2

# CUDA
if tc.cuda.is_available():
    device = tc.device("cuda")
else:
    device = tc.device("cpu")

# print(f"CUDA is available: {tc.cuda.is_available()}")


#### MODEL CLASSES ####
#######################

class Quickdraw_Dataset(Dataset):
    def __init__(self, ID_list, datapath, label_list_txt):

        self.ID_list = ID_list

        self.datapath = datapath

        self.label_list_txt = label_list_txt

        with open(label_list_txt, 'r') as f:
            self.label_list = [col.split('_')[0] for col in f.readlines()]

        with open(label_list_txt, 'r') as f:
            self.label_n_items_list = [int(col.split('_')[1].replace('\n', '')) for col in f.readlines()]
        self.label_n_items_list.insert(0,0)

        self.label_n_items_list_cut = self.label_n_items_list[1:len(self.label_n_items_list)]

        self.label = tc.zeros(len(self.label_list), dtype = tc.float32)

    def __len__(self):
        return len(self.ID_list)

    def get_line_from_txt(self, line_idx, txt_path):
        with open(txt_path, 'r') as f:
            line = itertools.islice(f, line_idx, line_idx+1)
            line = map(lambda s: s.strip(), line)
            return list(line)[0]
    
    def split_trn_tst(self, trn_tst_ratio, seed=1):
        self.rnd_ID_list = self.ID_list.copy()
        np.random.seed(seed)
        np.random.shuffle(self.rnd_ID_list)

        self.ID_list_trn = self.rnd_ID_list[0 : math.floor(len(self.ID_list) * trn_tst_ratio)]
        self.ID_list_tst = self.rnd_ID_list[math.ceil(len(self.ID_list) * trn_tst_ratio) : len(self.ID_list)]

        del self.rnd_ID_list
        np.random.seed()

        return Quickdraw_Dataset(self.ID_list_trn, self.datapath, self.label_list_txt), Quickdraw_Dataset(self.ID_list_tst, self.datapath, self.label_list_txt)

    def __getitem__(self, index):
        index = self.ID_list[index]
        label_idx = 0
        for n_items in self.label_n_items_list_cut:
            if n_items > index:
                category = self.label_list[label_idx]
                index = index - self.label_n_items_list[label_idx]
                filepath = self.get_line_from_txt(index, f"{self.datapath}/{category}/address_list.txt")
                break
            label_idx += 1

        sample = tv.transforms.ToTensor()(Image.open(filepath))
        
        label_cpy = self.label.clone()
        label_cpy[label_idx] = 1.

        return sample, label_cpy
    

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.ReLU()
        self.lin3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.lin1(x)
        out = self.lin2(out)
        out = self.lin3(out)
        return out
    

class NewConvNN(nn.Module):
    # Conv Output Size:
    # OutputWidth = (Width - FilterSize + 2*Padding) / (Stride) + 1
    def conv2d_out_dim(self, input_dim, kernel_size, padding, stride):
        return ((tc.tensor(input_dim) - tc.tensor(kernel_size) + 2*tc.tensor(padding)) / (tc.tensor(stride))) + 1
        # return (input_dim - kernel_size + 2*padding) / (stride) + 1
    
    def __init__(self, img_dim, fc1_dim, fc2_dim, fc3_dim, output_dim):
        super(ConvNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2,
                                 padding = 0)
        

        self.conv1 = nn.Conv2d(in_channels = img_dim[0],
                               out_channels = 2,
                               kernel_size = 9,
                               padding = 0,
                               stride = 1
                                )
        self.adapter_dim = self.conv2d_out_dim(img_dim[1:len(img_dim)], self.conv1.kernel_size, self.conv1.padding, self.conv1.stride)
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.pool.kernel_size, self.pool.padding, self.pool.stride)

        # self.conv2 = nn.Conv2d(in_channels = self.conv1.out_channels,
        #                        out_channels = 2,
        #                        kernel_size = 9,
        #                        padding = 0,
        #                        stride = 1
        #                         )
        # self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.conv2.kernel_size, self.conv2.padding, self.conv2.stride)
        # self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.pool.kernel_size, self.pool.padding, self.pool.stride)
        
  
        self.adapter_dim = int((self.conv1.out_channels * self.adapter_dim[0] * self.adapter_dim[1]).item())
        print(self.adapter_dim)

        self.fc1 = nn.Linear(in_features = self.adapter_dim, out_features = output_dim)


        # self.fc1 = nn.Linear(in_features = self.adapter_dim, out_features = fc1_dim)
        # self.fc4 = nn.Linear(in_features = self.fc3.out_features, out_features = output_dim)
        
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))

        out = out.view(-1, self.adapter_dim)
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class ConvNN(nn.Module):

    # Conv Output Size:
    # OutputWidth = (Width - FilterSize + 2*Padding) / (Stride) + 1
    def conv2d_out_dim(self, input_dim, kernel_size, padding, stride):
        return ((tc.tensor(input_dim) - tc.tensor(kernel_size) + 2*tc.tensor(padding)) / (tc.tensor(stride))) + 1
        # return (input_dim - kernel_size + 2*padding) / (stride) + 1

    def __init__(self, img_dim, fc1_dim, fc2_dim, fc3_dim, output_dim):
        super(ConvNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2,
                                 padding = 0)
        

        self.conv1 = nn.Conv2d(in_channels = img_dim[0],
                               out_channels = 8,
                               kernel_size = 9,
                               padding = 0,
                               stride = 1
                                )
        self.adapter_dim = self.conv2d_out_dim(img_dim[1:len(img_dim)], self.conv1.kernel_size, self.conv1.padding, self.conv1.stride)
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.pool.kernel_size, self.pool.padding, self.pool.stride)


        self.conv2 = nn.Conv2d(in_channels = self.conv1.out_channels,
                               out_channels = 8,
                               kernel_size = 9,
                               padding = 0,
                               stride = 1
                                )
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.conv2.kernel_size, self.conv2.padding, self.conv2.stride)
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.pool.kernel_size, self.pool.padding, self.pool.stride)

        self.conv3 = nn.Conv2d(in_channels = self.conv2.out_channels,
                               out_channels = 8,
                               kernel_size = 9,
                               padding = 0,
                               stride = 1
                                )
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.conv3.kernel_size, self.conv3.padding, self.conv3.stride)
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.pool.kernel_size, self.pool.padding, self.pool.stride)
  
        self.adapter_dim = int((self.conv3.out_channels * self.adapter_dim[0] * self.adapter_dim[1]).item())
        print(self.adapter_dim)

        self.fc1 = nn.Linear(in_features = self.adapter_dim, out_features = fc1_dim)
        self.fc2 = nn.Linear(in_features = self.fc1.out_features, out_features = fc2_dim)
        self.fc3 = nn.Linear(in_features = self.fc2.out_features, out_features = fc3_dim)
        self.fc4 = nn.Linear(in_features = self.fc3.out_features, out_features = output_dim)
        
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))

        out = out.view(-1, self.adapter_dim)
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


###### FUNCTIONS ######
#######################

def loading_animation(event, message = 'loading'):                  # Thread, needs a halting event as input!
    while(True):                                                    # prints a loading animation, run this code somewhere to see what it does.
        print(message, sep='', end='')                              # google "python threads" if you are unfamiliar.
        time.sleep(1)
        if event.is_set():
            clear_output(wait=True)
            break
        for i in range(3):
            print('.', sep='', end='')
            time.sleep(1)
            if event.is_set():
                clear_output(wait=True)
                break
        clear_output(wait=True)


def get_batch(batches,                              # input dataset, batch-length must be > 0
              batch_idx = 0,                        # get batch at batch_idx...
              get_random = False):                  # ...or pick a random batch
    
    # Start loading screen
    # event = Event()
    # thread = Thread(target = loading_animation, daemon=True, args=(event, "Loading Batches. This might take a while"))
    # thread.start()

    batches_iterator = iter(batches)                                            # init dataset iterator
    if get_random:                                                              # set random index
        batch_idx = random.randrange(len(batches))
    else:                                                                       # for proper indexing
        batch_idx += 1

    for i in range(batch_idx):                                                  # iterate through dataset...
        next(batches_iterator)
        if i == batch_idx - 1:                                                  # ...until idx...
            batch_spl, batch_lbl = next(batches_iterator)

            # Stop loading screen
            # if not event.is_set():
            #     event.set()
            #     thread.join()
            
            return (batch_spl, batch_lbl)                                       # ...and return a tuple of form (batch of samples, batch of labels)


def get_sample(batches,                             # input dataset, batch-length must be > 0
               sample_idx = 0,                      # get sample at sample_idx,...
               get_random = False):                 # ...or pick a random sample throughout all batches
    
    batch_idx, sample_idx = divmod(sample_idx, len(batches))
    batch = get_batch(batches, batch_idx, get_random)
    if get_random:
        sample_idx = random.randrange(len(batch)) 
    return (batch[0][sample_idx], batch[1][sample_idx])                         # returns a tuple of form (sample, label)
    

def tb_write_model(model, database):
    logdir =  './runs/'
    writer = SummaryWriter(logdir)
    writer.add_graph(model, database.__getitem__(0)[0])

    writer.flush()
    writer.close()

def tb_analytics_block(dir_counter):
    logdir = f"./runs/{dir_counter}/"
    writer = SummaryWriter(logdir)
    # tb_write_model(model, quickdraw_trn)
    dir_counter += 1
    return logdir, writer, dir_counter

logdir, writer, dir_counter = tb_analytics_block(dir_counter)


def training_loop(model,                            # model input
                  batches_trn,                      # training batches input
                  criterion,                        # cost/loss/criterion function input
                  optimizer,                        # ...
                  scheduler,
                  n_epochs = 1,                     # number of iterations through all batches
                  tb_analytics = False,             # tensorboard plugin
                  print_fps = 30.):                 # output fps. Needed for not overwhelming the kernel. Also serves to limit tensorboard datasize.
    
    # Start loading screen
    # event = Event()
    # thread = Thread(target = loading_animation, daemon=True, args=(event, "Loading Batches. This might take a while"))
    # thread.start()

    # init func-global variables
    model = model.to(device)
    criterion = criterion.to(device)

    n_batches = len(batches_trn)                                                    # number of batches
    acc = 0.                                                                        # accuracy counter
    n_iter = 0                                                                      # iter counter
    t_0 = time.time()                                                               # save current time value
    t_fps = time.time()                                                             # -- " -- for output fps
    
    for epoch in range(n_epochs):                                                   # iter through epochs
        for batch_idx, (inputs, labels) in enumerate(batches_trn):                  # iter through batches in epoch
            
            # Stop loading screen
            # if not event.is_set():
            #     event.set()
            #     thread.join()
            
            # Compute prediction and true value Block
            output_prd = model(inputs.to(device))                                   # calc model output
            output_tru = labels.to(device)                                          # get label
            
            # Running average accuracy Block
            compare_results = 0
            for outpt, label in zip(output_prd, output_tru):                        # calc number of correct predictions in batch
                compare_results += outpt.argmax() == label.argmax()

            batch_len = len(labels)                                                 # get size of current batch. Not necessarily equal to batch_size!
            acc = (compare_results + n_iter * acc) / (n_iter + batch_len)           # calc running average accuracy
            n_iter += batch_len                                                     # advance iteration counter

            # Gradient Block
            optimizer.zero_grad()                                                   # reset gradient calc
            cost = criterion(output_prd, output_tru)                                # calc cost value
            cost.backward()                                                         # backward propagation
            optimizer.step()                                                        # apply optimizer
            
            # Output Block
            if time.time() - t_fps >= 1./print_fps:
                t_fps = time.time()
                print(f'epoch {epoch + 1}/{n_epochs}; batch {batch_idx + 1}/{n_batches}; learning rate = {optimizer.param_groups[0]["lr"]}; Cost: {cost:.6f}; Running Accuracy: {100 * acc:.2f} %')
                clear_output(wait=True)
            # Define your tensorboard data here 
                if tb_analytics:
                    writer.add_scalar('Training Loss', cost, batch_idx + epoch * n_batches)
                    writer.add_scalar('Training Accuracy', acc, batch_idx + epoch * n_batches)
            
            # tidy up tensorboard writer
            if tb_analytics:
                writer.flush()
                writer.close()
            
        scheduler.step()                                                            # diminish learning rate after every epoch
    
    t_1 = time.time()                                                               # get time value after training
    print(f'Done. Final Cost: {cost:.6f}. Time: {(t_1 - t_0):.2f}s.')               # final output
    return model                                                                    # returns the trained model
    

def validation_loop(model,                                                          # model input
                    batches_tst,                                                    # test batches input
                    print_miss = False,                                             # option for showing wrong predictions
                    print_fps = 30.):                                               # output fps. Needed for not overwhelming the kernel
    
    with tc.no_grad():                                                              # don't train the model anymore!

        # Start loading screen
        # event = Event()
        # thread = Thread(target = loading_animation, daemon=True, args=(event, "Loading Batches. This might take a while"))
        # thread.start()

        model = model.to(device)
        
        # init func-global variables
        n_batches = len(batches_tst)                                                # number of batches
        acc = 0.                                                                    # running accuracy counter
        n_iter = 0                                                                  # iter counter
        t_0 = time.time()                                                           # get current time value
        t_fps = time.time()                                                         # -- " -- for output fps
    
        for batch_idx, (inputs, labels) in enumerate(batches_tst):                  # iter through batches
            
            # Stop loading screen
            # if not event.is_set():
            #     event.set()
            #     thread.join()
            
            for inpt, label in zip(inputs, labels):                                 # iter through samples in batches
                
                output_prd = model(inpt.to(device))                                 # calc model output
                output_tru = label.to(device)                                       # get label
                compare_results = output_prd.argmax() == output_tru.argmax()
                acc = (compare_results + n_iter * acc) / (1 + n_iter)               # running average accuracy
                n_iter += 1

                # Print miss Block
                if print_miss and not compare_results:
                    print(f'batch {batch_idx + 1}/{n_batches}; Accuracy: {100*acc:.2f} %')
                    print(f'Miss at Iteration {n_iter}! Predicted: {output_prd.argmax().item()} (Confidence = {100 * tc.softmax(output_prd, dim=0).max().item():.2f} %), True: {output_tru.argmax().item()}')
                    plt.imshow(inpt.reshape(28,28), cmap='gray')
                    plt.axis('off')
                    plt.show()
                    time.sleep(2)
                    clear_output(wait=True)  
                # Output Block
                if time.time() - t_fps >= 1./print_fps:
                    t_fps = time.time()
                    print(f'batch {batch_idx + 1}/{n_batches}; Accuracy: {100*acc:.2f} %')
                    clear_output(wait=True)

                    
        print(f'Done. Final Accuracy: {100*acc:.2f} %. Time: {(time.time() - t_0):.2f}s.')  # final output
    return acc                                                                      # returns the final accuracy
