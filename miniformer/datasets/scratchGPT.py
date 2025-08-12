import os

import torch
from torch.utils.data import Dataset

class ScratchGPTDataset(Dataset):

    def __init__(self, filepath, tokenizer, config):

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                raw_text = f.read()
        else:
            raise FileNotFoundError("Filepath does not exist:", filepath)

        print("Length of text:", len(raw_text))  

        input_ids = tokenizer.encode(raw_text)
        self.context, self.target = [], []
        for index in range(0, len(input_ids) - config.context_length, config.stride):

            context = input_ids[index: index + config.context_length]
            target = input_ids[index + 1: index + config.context_length + 1]

            self.context.append(torch.LongTensor(context))
            self.target.append(torch.LongTensor(target))

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        return self.context[index], self.target[index]