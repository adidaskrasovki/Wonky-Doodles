### ALL THE IMPORTS ###
#######################

# PyTorch imports
import torch as tc
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# Standard pkg imports
import os

from RecognitionNN_utils import *

if __name__ == "__main__":

    # CUDA
    if tc.cuda.is_available():
        device = tc.device("cuda")
    else:
        device = tc.device("cpu")

    print(f"CUDA is available: {tc.cuda.is_available()}")


#### PREPARE DATA ####
###### - 0.1 - #######

    # Adapter from your Dataset to Dataloader and thus specific for each Dataset.
    # Microsoft Defender might slow things down here significantly. Take a look at your task manager.

    datapath = "E:/WonkyDoodles/qd_png"
    train_test_ratio = 0.99                                     # define ratio of train/total samples


    ID_list = []
    ID = 0
    for dir in os.listdir(datapath):
            with open(f"{datapath}/{dir}/address_list.txt", 'r') as g:
                    for line in g:
                        ID_list.append(ID)
                        ID += 1

    quickdraw_dataset = Quickdraw_Dataset(ID_list, datapath, "label_list.txt")
    quickdraw_trn, quickdraw_tst = quickdraw_dataset.split_trn_tst(train_test_ratio, seed=3)
    
    del ID
    del ID_list
    del quickdraw_dataset

##### DATALOADER #####
###### - 0.2 - #######

    # Shove both training and test lists into the dataloader.

    batch_size = 64                                                 # define batch_size

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

    model = ConvNN(quickdraw_trn.__getitem__(0)[0].shape,
                fc1_dim = int(512),
                fc2_dim = int(256),
                fc3_dim = int(128),
                output_dim = len(quickdraw_trn.label_list)
                ).to(device)
    

    # Optional Block for loading in a model and/or model-state from disk
    filepath = './models/'
    filename = "Wonky_Doodles_CNN2_FFN3_lite50.pth"
    # model = tc.load(f"{filepath}{filename}", map_location=device)
    model.load_state_dict(tc.load(f"{filepath}{filename.replace('.pth', '')}_state_dict.pth", map_location=device))

###### LOSS/COST ######
###### OPTIMIZER ######
###### SCHEDULER ######
######## - 2 - ########

    criterion = CrossEntropyLoss()

    optimizer = tc.optim.Adam(model.parameters(),
                            lr = .001)                                     # define initial learning rate
    
    # Optional Block for loading in a optimizer-state from disk
    # filepath = './models/'
    # filename = "Wonky_Doodles_CNN2_FFN3_lite10.pth"
    optimizer.load_state_dict(tc.load(f"{filepath}{filename.replace('.pth', '')}_state_dict_optim.pth", map_location=device))
    optimizer.param_groups[0]["lr"] = .0001

    step_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                            step_size = 1,                  # diminish learning rate every n-th epoch...
                                            gamma = 1.)                     # ...by this diminishing factor

    # Optional Block for loading in a lr-scheduler-state from disk
    # filepath = './models/'
    # filename = "Wonky_Doodles_CNN2_FFN3_lite10.pth"
    step_lr_scheduler.load_state_dict(tc.load(f"{filepath}{filename.replace('.pth', '')}_state_dict_lr_scheduler.pth", map_location=device))

###### TRAINING #######
######## - 3 - ########

    # Optional Block for using tensorboard
    tb_analytics = False


    # Training Loop. 
    # Note that loading the batches into memory might take a few minutes. Take a look at your task manager.
    model_trn = training_loop(n_epochs = 1,                             # See func definition for input details
                            print_fps = 5.,
                            model = model,
                            batches_trn = batches_trn,
                            criterion = criterion,
                            optimizer = optimizer,
                            scheduler = step_lr_scheduler,
                            device = device)


##### SAVE MODEL ######
######## - 4 - ########

    # Set filepath and model name
    filepath = './models/'
    model_name = 'Wonky_Doodles_CNN2_FFN3_liteXX'

    # tc.save(model_trn, f"{filepath}{model_name}.pth")                               # Save the whole model and/or...
    # tc.save(optimizer, f"{filepath}{model_name}_optim.pth")
    # tc.save(step_lr_scheduler, f"{filepath}{model_name}_lr_scheduler.pth")
    tc.save(model_trn.state_dict(), f"{filepath}{model_name}_state_dict.pth")       # ...save only the the model state
    tc.save(optimizer.state_dict(), f"{filepath}{model_name}_state_dict_optim.pth")
    tc.save(step_lr_scheduler.state_dict(), f"{filepath}{model_name}_state_dict_lr_scheduler.pth")

    

    print(f"Saved Model to {filepath}{model_name}.")


####### TESTING #######
######## - 5 - ########

    # Testing Loop.
    # Again, loading batches into memory may take a few minutes. Stay strong.

    accuracy = validation_loop(print_fps = 5.,                          # See func definition for input details
                            print_miss = False,
                            model = model_trn.to(device),
                            batches_tst = batches_tst,
                            device = device)
