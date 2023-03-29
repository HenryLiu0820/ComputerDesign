import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import tqdm
import os
from dataloader import *
import argparse

def predict(model, dataset, device):    # the dataset here has to be a torch.utils.data.Dataset object
    model.eval()

    preds = []
    for batch in tqdm(dataset):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            X, attention_mask = batch
            output = model(X, attention_mask=attention_mask)
            logits = output.logits
            preds.extend(logits.cpu().tolist())

    return preds

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/prediction.py', type=str, help='path to local model')
    parser.add_argument('--dataset_name', default='', type=str, help='name of dataset to be used')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = set_args()
    # load model from local file
    model_path = args.model_path
    dataset_name = args.dataset_name  # the path to the dataset, the dataset is a .csv file stored in the ../data directory
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    dataset = SentimentDataset(tokenizer, [dataset_name])

    if os.path.exists(model_path) == True:
        model = torch.load(model_path)
        preds = predict(model, dataset, 'cuda')


