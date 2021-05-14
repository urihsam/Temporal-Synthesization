import os
import io
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils.utils import OrderedCounter

class EHR(Dataset):

    def __init__(self, data_dir, split, split_ratio=0.8, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.max_length = kwargs.get('max_length', None)

        data_file = os.path.join(data_dir, 'data.npy')
        data_len_file = os.path.join(data_dir, 'data_len.npy')

        # check whether has been splitted or not
        assert split in ["train", "valid", "test"]
        if not os.path.exists(os.path.join(data_dir, split)):
            data = np.load(data_file, allow_pickle=True)
            data_length = np.load(data_len_file, allow_pickle=True)

            if self.max_length == None:
                self.max_length = max(data_length)
            
            total_size = len(data_length)
            train_size = int(total_size * split_ratio)
            valid_size = (total_size - train_size) // 2
            test_size = total_size - train_size - valid_size

            total_idx = list(range(len(data)))
            random.shuffle(total_idx)
            train_idx = total_idx[:train_size]
            valid_idx = total_idx[train_size:train_size+valid_size]
            test_idx = total_idx[train_size+valid_size:]

            train_data = self.load_data(data, train_idx)
            valid_data = self.load_data(data, valid_idx)
            test_data = self.load_data(data, test_idx)

            # save
            train_path = os.path.join(data_dir, "train"); os.mkdir(train_path)
            np.save(os.path.join(train_path, "data.npy"), train_data)
            valid_path = os.path.join(data_dir, "valid"); os.mkdir(valid_path)
            np.save(os.path.join(valid_path, "data.npy"), valid_data)
            test_path = os.path.join(data_dir, "test"); os.mkdir(test_path)
            np.save(os.path.join(test_path, "data.npy"), test_data)

        path = os.path.join(data_dir, split)
        self.data = np.load(os.path.join(path, "data.npy").format(split), allow_pickle=True)

            
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        
        return {
            "tempo": np.asarray(self.data[idx]["tempo"]),
            "target": np.asarray(self.data[idx]["target"]),
            "mask": np.asarray(self.data[idx]["mask"]),
            "target_mask": np.asarray(self.data[idx]["target_mask"])
        }

 
    def load_data(self, data, index):
        examples = []
        for idx in index:
            ex = data[idx]
            struc = ex["struc"]
            mask = ex["mask"]
            tempo = ex["tempo"]
            # add struc into tempo
            padding = np.array([0.0, 0.0])
            struc = [padding] + struc
            extra = np.expand_dims(np.concatenate(struc, 0), 0)
            extra_mask = np.expand_dims(np.array([0,0]+[1]*7), 0)
            tempo = np.concatenate([extra, tempo], 0)
            mask = np.concatenate([extra_mask, mask], 0)
            # cut or padding
            padding = np.expand_dims(np.array([0.0]*9), 0)
            if tempo.shape[0] > self.max_length:
                tempo = tempo[:self.max_length, :]
                target = np.concatenate([tempo[1:, :], tempo[[-1], :]], 0)
                mask = mask[:self.max_length, :]
                target_mask = np.concatenate([mask[1:, :], mask[[-1], :]], 0)
            else:
                add = self.max_length - tempo.shape[0]
                target = np.concatenate([tempo[1:, :], tempo[[-1], :]]+[padding]*add, 0)
                tempo = np.concatenate([tempo]+[padding]*add, 0)
                target_mask = np.concatenate([mask[1:, :], mask[[-1], :]]+[padding]*add, 0)
                mask = np.concatenate([mask]+[padding]*add, 0)
                


            # append
            example = {
                "tempo": tempo,
                "target": target,
                "mask": mask,
                "target_mask": target_mask
            }
            examples.append(example)

        return examples

    
    '''
    @property
    def train_size(self):
        return self.train_size
    
    @property
    def valid_size(self):
        return self.valid_size
    
    @property
    def test_size(self):
        return self.test_size

    
    def shuffle(self):
        self.train_shuffle()
        self.valid_shuffle()
        self.test_shuffle()


    def train_shuffle(self):
        index = list(range(self.train_size))
        random.shuffle(index)
        self.train_data = self.train_data[index]
        self.train_mask = self.train_mask[index]
        self._train_batch_idx = 0
    
 
    def valid_shuffle(self):
        index = list(range(self.valid_size))
        random.shuffle(index)
        self.valid_data = self.valid_data[index]
        self.valid_mask = self.valid_mask[index]
        self._valid_batch_idx = 0
    
    
    def test_shuffle(self):
        index = list(range(self.test_size))
        random.shuffle(index)
        self.test_data = self.test_data[index]
        self.test_mask = self.test_mask[index]
        self._test_batch_idx = 0

    
    def next_train_batch(self, batch_size): 
        if self._train_batch_idx+batch_size > self.train_size:
            self.train_shuffle()
        batch_data = self.train_data[self._train_batch_idx: self._train_batch_idx+batch_size]
        batch_mask = self.train_mask[self._train_batch_idx: self._train_batch_idx+batch_size]
        self._train_batch_idx += batch_size
        return batch_data, batch_mask


    def next_valid_batch(self, batch_size): 
        if self._valid_batch_idx+batch_size > self.valid_size:
            self.valid_shuffle()
        batch_data = self.valid_data[self._valid_batch_idx: self._valid_batch_idx+batch_size]
        batch_mask = self.valid_mask[self._valid_batch_idx: self._valid_batch_idx+batch_size]
        self._valid_batch_idx += batch_size
        return batch_data, batch_mask


    def next_test_batch(self, batch_size): 
        if self._test_batch_idx+batch_size > self.test_size:
            self.test_shuffle()
        batch_data = self.test_data[self._test_batch_idx: self._test_batch_idx+batch_size]
        batch_mask = self.test_mask[self._test_batch_idx: self._test_batch_idx+batch_size]
        self._test_batch_idx += batch_size
        return batch_data, batch_mask
    '''