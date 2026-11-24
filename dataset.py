from torch.utils.data import Dataset
import torch


class GPTDataset(Dataset):
    def __init__(self, ids, context_length):
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.ids) - self.context_length

    def __getitem__(self, index):
        x = self.ids[index: index + self.context_length]
        y = self.ids[index + 1: index + self.context_length + 1]

        return x, y
