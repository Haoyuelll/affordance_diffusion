import os
import numpy as np

def get_suffix_length(fname:str):
    sub_id = fname.find("frame")
    suff_len = len(fname) - sub_id - fname[sub_id:].find("_")
    return suff_len

data_root = "vlr_data/test_mask"
flist = os.listdir(data_root)
suff_id = -1*get_suffix_length(flist[0])
split = [fname[:suff_id] for fname in flist]
        
np.random.shuffle(split)

val_prop = 0.1
split_id = int((1-val_prop)*len(split))

train_split = "\n".join(split[:split_id])
val_split = "\n".join(split[split_id:])


parental_root = data_root.split("/")[0]
with open(os.path.join(parental_root, "train_split.txt"), "w") as ft:
    ft.write(train_split)
with open(os.path.join(parental_root, "val_split.txt"), "w") as fv:
    ft.write(val_split)
    