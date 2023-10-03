from RecognitionNN_utils import Quickdraw_Dataset
import os
import torchvision.transforms as T

#### PREPARE DATA ####
###### - 0.1 - #######
if __name__ == "__main__":

    # Adapter from your Dataset to Dataloader and thus specific for each Dataset.
    # Microsoft Defender might slow things down here significantly. Take a look at your task manager.

    datapath = "E:/WonkyDoodles/qd_png"
    train_test_ratio = 0.98                                     # define ratio of train/total samples
    start_ratio = 0.
    end_ratio = 1.


    ID_list = []
    ID = 0
    for dir in os.listdir(datapath):
            with open(f"{datapath}/{dir}/address_list.txt", 'r') as g:
                    # n_lines = len(g)
                    for line in g:
                        # if idx > end_ratio * n_lines: break
                        ID_list.append(ID)
                        ID += 1

qd = Quickdraw_Dataset(ID_list, datapath, "label_list.txt")

n_lines = 0
with open(f'{datapath}/aircraft carrier/address_list.txt', 'r') as f:
      for line in f:
            n_lines += 1

img = T.ToPILImage()(qd.__getitem__(9459)[0])

print(n_lines)
img.show()