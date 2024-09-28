import json

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
# from data.dataset import Multi_Mod_Dataset
    
class Data(Dataset):
    def __init__(self):
        with open('/remote-home/share/SAM/trainset.jsonl', 'r') as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        data_2d = []
        data_3d = []
        for sample in lines:
            if sample['is_3D']=='3D':
                data_3d.append(sample)
            elif sample['is_3D']=='2D':
                data_2d.append(sample)
        self.data_split = {'2d':[0, len(data_2d)], '3d':[len(data_2d), len(data_2d)+len(data_3d)]}
        self.datasets_samples = data_2d + data_3d

    def __getitem__(self, index):
        return  self.datasets_samples[index]['dataset']

    def __len__(self):
        return len(self.datasets_samples)
    
class BatchSampler:
    def __init__(self, dataset_split, batch_size, drop_last=True):
        """
        Rearrange data indexes, divide 2D and 3D samples into different batches
        
        Args:
            dataset_split (dict): a dict indicating index ranges of different types of samples, e.g. {'3d':[start, end], ......} 
            NOTE: include start, exclude end
            batch_size (dict): a dict indicating batch_size of 2D and 3D samples, {'2D':128, '3D':8}
            drop_last (bool): default as Ture
        """
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self.rearrange_index()
        
    def rearrange_index(self):
        """
        Randomly divide samples into batches, boundary between two batches:
        1. the data type is changed, i.e. is_3d
        2. len of the batch has reached batch_size of 2d or 3d
        """
        # generate indexes for 2d and 3d samples and randomly shuffle (actually unnecessary if Dataset shuffle samples before every epoch
        sample_2d_ls = []
        sample_3d_ls = []
        for split, index_span in self.dataset_split.items():
            if '2d' in split:
                sample_2d_ls += [x for x in range(index_span[0], index_span[1])]
            elif '3d' in split:
                sample_3d_ls += [x for x in range(index_span[0], index_span[1])]
            else:
                raise ValueError(f'Data type (2D or 3D) are not specified in dataset split {split}.')
        random.shuffle(sample_2d_ls)
        random.shuffle(sample_3d_ls)
        # define the sequence of 2d and 3d batches (ensure randomness of batch sequence
        batch_2d_num = len(sample_2d_ls) // self.batch_size['2D']
        batch_3d_num = len(sample_3d_ls) // self.batch_size['3D']
        if not self.drop_last:
            batch_2d_num += 1
            batch_3d_num += 1
        self.batch_2d_3d_seq = [0] * batch_2d_num + [1] * batch_3d_num
        random.shuffle(self.batch_2d_3d_seq)
        # fill in sample indexes for each batch
        cursor_2d = 0
        cursor_3d = 0
        self.index_ls = []
        self.is_3d = []
        for is_3d in self.batch_2d_3d_seq:
            if is_3d:
                self.index_ls += sample_3d_ls[cursor_3d : min(cursor_3d+self.batch_size['3D'], len(sample_3d_ls))]
                self.is_3d += ['3D'] * min(self.batch_size['3D'], len(sample_3d_ls)-cursor_3d)
                cursor_3d += self.batch_size['3D']
            else:
                self.index_ls += sample_2d_ls[cursor_2d : min(cursor_2d+self.batch_size['2D'], len(sample_2d_ls))]
                self.is_3d += ['2D'] * min(self.batch_size['2D'], len(sample_2d_ls)-cursor_2d)
                cursor_2d += self.batch_size['2D']
    
    def __iter__(self):
        batch = []
        for i, idx in enumerate(self.index_ls):
            batch.append(idx)
            if len(batch) == self.batch_size[self.is_3d[i]]:   # 凑齐一个batch了
                yield batch
                batch = []

            if (i < len(self.is_3d) - 1 and self.is_3d[i] != self.is_3d[i+1]):  # 下一个data切换2d/3d，即构成新的batch
                if len(batch) > 0:
                    assert self.drop_last == False
                    yield batch
                    batch = []
                else:
                    batch = []

        if len(batch) > 0:
            assert self.drop_last == False
            yield batch

    def __len__(self):
        return len(self.batch_2d_3d_seq)


class BatchSampler_noshuffle:
    def __init__(self, dataset_split, batch_size, drop_last=True):
        """
        Rearrange data indexes, divide 2D and 3D samples into different batches
        
        Args:
            dataset_split (dict): a dict indicating index ranges of different types of samples, e.g. {'3d':[start, end], ......} 
            NOTE: include start, exclude end
            batch_size (dict): a dict indicating batch_size of 2D and 3D samples, {'2D':128, '3D':8}
            drop_last (bool): default as Ture
        """
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self.rearrange_index()
        
    def rearrange_index(self):
        """
        Randomly divide samples into batches, boundary between two batches:
        1. the data type is changed, i.e. is_3d
        2. len of the batch has reached batch_size of 2d or 3d
        """
        # generate indexes for 2d and 3d samples and randomly shuffle (actually unnecessary if Dataset shuffle samples before every epoch
        sample_2d_ls = []
        sample_3d_ls = []
        for split, index_span in self.dataset_split.items():
            if '2d' in split:
                sample_2d_ls += [x for x in range(index_span[0], index_span[1])]
            elif '3d' in split:
                sample_3d_ls += [x for x in range(index_span[0], index_span[1])]
            else:
                raise ValueError(f'Data type (2D or 3D) are not specified in dataset split {split}.')
        # random.shuffle(sample_2d_ls)
        # random.shuffle(sample_3d_ls)
        # define the sequence of 2d and 3d batches (ensure randomness of batch sequence
        batch_2d_num = len(sample_2d_ls) // self.batch_size['2D']
        batch_3d_num = len(sample_3d_ls) // self.batch_size['3D']
        if not self.drop_last:
            batch_2d_num += 1
            batch_3d_num += 1
        self.batch_2d_3d_seq = [0] * batch_2d_num + [1] * batch_3d_num
        # random.shuffle(self.batch_2d_3d_seq)
        # fill in sample indexes for each batch
        cursor_2d = 0
        cursor_3d = 0
        self.index_ls = []
        self.is_3d = []
        for is_3d in self.batch_2d_3d_seq:
            if is_3d:
                self.index_ls += sample_3d_ls[cursor_3d : min(cursor_3d+self.batch_size['3D'], len(sample_3d_ls))]
                self.is_3d += ['3D'] * min(self.batch_size['3D'], len(sample_3d_ls)-cursor_3d)
                cursor_3d += self.batch_size['3D']
            else:
                self.index_ls += sample_2d_ls[cursor_2d : min(cursor_2d+self.batch_size['2D'], len(sample_2d_ls))]
                self.is_3d += ['2D'] * min(self.batch_size['2D'], len(sample_2d_ls)-cursor_2d)
                cursor_2d += self.batch_size['2D']
    
    def __iter__(self):
        batch = []
        for i, idx in enumerate(self.index_ls):
            batch.append(idx)
            if len(batch) == self.batch_size[self.is_3d[i]]:   # 凑齐一个batch了
                yield batch
                batch = []

            if (i < len(self.is_3d) - 1 and self.is_3d[i] != self.is_3d[i+1]):  # 下一个data切换2d/3d，即构成新的batch
                if len(batch) > 0:
                    assert self.drop_last == False
                    yield batch
                    batch = []
                else:
                    batch = []

        if len(batch) > 0:
            assert self.drop_last == False
            yield batch

    def __len__(self):
        return len(self.batch_2d_3d_seq)


if __name__ == '__main__':
    d = Data()
    data_split = d.data_split
    batch_size = {'3D':8, '2D':16}
    bs = BatchSampler(data_split, batch_size, False)
    dl = DataLoader(d, batch_sampler=bs)
    
    count = 0
    for b in dl:
        print(b)
        count += 1
        if count == 3:
            break

    # tensor([ 9,  8, 11,  7])
    # tensor([3, 2, 0])
    # tensor([ 4,  6, 10,  5])
    # tensor([1]) 