from dotenv import load_dotenv

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from .model import ScratchGPTLMModel
from ...datasets.scratchGPT import ScratchGPTDataset
from ...config import Config

load_dotenv(dotenv_path="/home/ssubrahmanya/gpt2/.env")
torch.manual_seed(42)

def get_model(config):

    tokenizer = tiktoken.get_encoding("gpt2")
    model = ScratchGPTLMModel(config)

    return tokenizer, model


def get_loader(config, tokenizer):

    train_dataset = ScratchGPTDataset(
      filepath = config.train,
      tokenizer = tokenizer,
      config = config)

    train_loader = DataLoader(
      dataset = train_dataset,
      batch_size = config.batch_size,
      shuffle = True,
      num_workers = 4)

    valid_dataset = ScratchGPTDataset(
      filepath = config.valid,
      tokenizer = tokenizer,
      config = config)

    valid_loader = DataLoader(
      dataset = valid_dataset,
      batch_size = config.batch_size,
      shuffle = False,
      num_workers = 4)

    return train_loader, valid_loader


def train(model, train_loader, valid_loader, criterion, optimizer, config):

    with wandb.init(project="miniFormer", config=config.to_dict()) as run:
    
        model.to(config.device)
        for epoch in range(config.epochs):

            model.train()
            train_loss = 0.0
            for batch_idx, (context, target) in enumerate(train_loader):

                context, target = context.to(config.device), target.to(config.device)

                logits = model(context)
                loss = criterion(logits.flatten(0, 1), target.flatten())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            run.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss
            })

            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for context, target in valid_loader:

                    context, target = context.to(config.device), target.to(config.device)

                    logits = model(context)
                    loss = criterion(logits.flatten(0, 1), target.flatten())
                    total_loss += loss.item()

                avg_loss = total_loss / len(valid_loader)
                run.log({
                    "epoch": epoch + 1,
                    "valid_loss": avg_loss
                })

            print("Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}".format(
                epoch + 1, config.epochs, avg_train_loss, avg_loss))
            
    return

if __name__ == "__main__":

    config = Config(
        train = "/home/ssubrahmanya/gpt2/miniformer/data/train.txt",
        valid = "/home/ssubrahmanya/gpt2/miniformer/data/test.txt",
        stride = 1,
        batch_size = 4,
        epochs = 10,
        vocab_size = 50257,
        context_length = 256,
        embedding_dim = 768,
        num_heads = 12,
        num_layers = 12,
        dropout = 0.2,
        qkv_bias = False, 
        device = "cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize tokenizer, model
    tokenizer, model = get_model(config)

    # get data loaders
    train_loader, valid_loader = get_loader(config, tokenizer)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # train the model
    train(model, train_loader, valid_loader, criterion, optimizer, config)