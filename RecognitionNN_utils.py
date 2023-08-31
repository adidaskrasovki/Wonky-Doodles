### ALL THE IMPORTS ###
#######################

# PyTorch imports
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

import torchvision as tv

# PIL Imports
from PIL import Image

# Standard pkg imports
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from threading import Thread as Thread
from threading import Event as Event
import re
import itertools

from torch.utils.tensorboard import SummaryWriter
logdir =  './runs/'                 # dir in which to save run data
writer = SummaryWriter(logdir)      # init tensorboard data writer

# CUDA
if tc.cuda.is_available():
    device = tc.device("cuda")
else:
    device = tc.device("cpu")

# print(f"CUDA is available: {tc.cuda.is_available()}")


#### MODEL CLASSES ####
#######################

class Quickdraw_Dataset(Dataset):
    def __init__(self, ID_list, labels_list):
        self.ID_list = ID_list
        self.label_list = labels_list

    def __len__(self):
        return len(self.ID_list)

    def get_line_from_txt(self, line_idx, txt_path):
    # Maybe implement this solution in the future:
        with open(txt_path) as f:
            line = itertools.islice(f, line_idx, line_idx+1)
            line = map(lambda s: s.strip(), line)
            return list(line)[0]
    # with open(txt_path) as f:
    #     for idx, line in enumerate(f):
    #         if idx == line_idx:
    #             return line.rstrip()
    
    def split_trn_tst(self, trn_tst_ratio):
        self.rnd_ID_list = self.ID_list
        np.random.shuffle(self.rnd_ID_list)

        self.ID_list_trn = self.rnd_ID_list[0 : math.floor(len(self.ID_list) * trn_tst_ratio)]
        self.ID_list_tst = self.rnd_ID_list[math.ceil(len(self.ID_list) * trn_tst_ratio) : len(self.ID_list)]

        del self.rnd_ID_list

        return Quickdraw_Dataset(self.ID_list_trn, self.label_list), Quickdraw_Dataset(self.ID_list_tst, self.label_list)

    def __getitem__(self, index):
        filepath = self.get_line_from_txt(self.ID_list[index], 'address_list.txt')
        sample = tv.transforms.ToTensor()(Image.open(f"{filepath}"))
        
        label = tc.zeros(len(self.label_list), dtype = tc.float32) 
        for idx, category in enumerate(self.label_list):
            if re.sub('\d', '', filepath).replace('.\\quickdraw_dataset_png\\', '').replace('\\.png', '') == category:
                label[idx] = 1.
                break

        return sample, label
    

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
    

class ConvNN(nn.Module):

    # Conv Output Size:
    # OutputWidth = (Width - FilterSize + 2*Padding) / (Stride) + 1
    def conv2d_out_dim(self, input_dim, kernel_size, padding, stride):
        return ((tc.tensor(input_dim) - tc.tensor(kernel_size) + 2*tc.tensor(padding)) / (tc.tensor(stride))) + 1
        # return (input_dim - kernel_size + 2*padding) / (stride) + 1

    def __init__(self, img_dim, fc1_dim, fc2_dim, output_dim):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = img_dim[0],
                               out_channels = 3,
                               kernel_size = 5,
                                )

        self.conv2 = nn.Conv2d(in_channels = self.conv1.out_channels,
                               out_channels = 16,
                               kernel_size = 5,
                                )

        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)
        
        self.adapter_dim = self.conv2d_out_dim(img_dim[1:len(img_dim)], self.conv1.kernel_size, self.conv1.padding, self.conv1.stride)
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.pool.kernel_size, self.pool.padding, self.pool.stride)
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.conv2.kernel_size, self.conv2.padding, self.conv2.stride)
        self.adapter_dim = self.conv2d_out_dim(self.adapter_dim, self.pool.kernel_size, self.pool.padding, self.pool.stride)
        self.adapter_dim = int((self.conv2.out_channels * self.adapter_dim[0] * self.adapter_dim[1]).item())

        self.fc1 = nn.Linear(in_features = self.adapter_dim, out_features = fc1_dim)
        self.fc2 = nn.Linear(in_features = self.fc1.out_features, out_features = fc2_dim)
        self.fc3 = nn.Linear(in_features = self.fc2.out_features, out_features = output_dim)
        
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, self.adapter_dim)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


###### FUNCTIONS ######
#######################

def loading_animation(event, message = 'loading'):                  # Thread, needs a halting event as input!
    while(True):                                                    # prints a loading animation, run this code somewhere to see what it does.
        print(message, sep='', end='')                              # google "python threads" if you are unfamiliar.
        time.sleep(1)
        if event.is_set():
            clear_output()
            break
        for i in range(3):
            print('.', sep='', end='')
            time.sleep(1)
            if event.is_set():
                clear_output()
                break
        clear_output(wait = True)


def get_batch(batches,                              # input dataset, batch-length must be > 0
              batch_idx = 0,                        # get batch at batch_idx...
              get_random = False):                  # ...or pick a random batch
    
    # Start loading screen
    event = Event()
    thread = Thread(target = loading_animation, daemon=True, args=(event, "Loading Batches. This might take a while"))
    thread.start()

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
            if not event.is_set():
                event.set()
                thread.join()
            
            return (batch_spl, batch_lbl)                                       # ...and return a tuple of form (batch of samples, batch of labels)


def get_sample(batches,                             # input dataset, batch-length must be > 0
               sample_idx = 0,                      # get sample at sample_idx,...
               get_random = False):                 # ...or pick a random sample throughout all batches
    
    batch_idx, sample_idx = divmod(sample_idx, len(batches))
    batch = get_batch(batches, batch_idx, get_random)
    if get_random:
        sample_idx = random.randrange(len(batch)) 
    return (batch[0][sample_idx], batch[1][sample_idx])                         # returns a tuple of form (sample, label)
    

def tb_write_model(model, batches):
    logdir =  './runs/'
    writer = SummaryWriter(logdir)
    writer.add_graph(model, get_sample(batches)[0].to(device))

    writer.flush()
    writer.close()


def training_loop(model,                            # model input
                  batches_trn,                      # training batches input
                  criterion,                        # cost/loss/criterion function input
                  optimizer,                        # ...
                  scheduler,
                  n_epochs = 1,                     # number of iterations through all batches
                  tb_analytics = False,             # tensorboard plugin
                  print_fps = 30.):                 # output fps. Needed for not overwhelming the kernel. Also serves to limit tensorboard datasize.
    
    # Start loading screen
    event = Event()
    thread = Thread(target = loading_animation, daemon=True, args=(event, "Loading Batches. This might take a while"))
    thread.start()

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
            if not event.is_set():
                event.set()
                thread.join()
            
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
                clear_output(wait = True)
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
        event = Event()
        thread = Thread(target = loading_animation, daemon=True, args=(event, "Loading Batches. This might take a while"))
        thread.start()

        model = model.to(device)
        criterion = criterion.to(device)
        
        # init func-global variables
        n_batches = len(batches_tst)                                                # number of batches
        acc = 0.                                                                    # running accuracy counter
        n_iter = 0                                                                  # iter counter
        t_0 = time.time()                                                           # get current time value
        t_fps = time.time()                                                         # -- " -- for output fps
    
        for batch_idx, (inputs, labels) in enumerate(batches_tst):                  # iter through batches
            
            # Stop loading screen
            if not event.is_set():
                event.set()
                thread.join()
            
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
                    clear_output(wait = True)  
                # Output Block
                if time.time() - t_fps >= 1./print_fps:
                    t_fps = time.time()
                    print(f'batch {batch_idx + 1}/{n_batches}; Accuracy: {100*acc:.2f} %')
                    clear_output(wait = True)

                    
        print(f'Done. Final Accuracy: {100*acc:.2f} %. Time: {(time.time() - t_0):.2f}s.')  # final output
    return acc                                                                      # returns the final accuracy

print('Done.')