import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataloader import SentimentDataset
import argparse
import pandas as pd

def predict(model, val_loader, device):    # the dataset here has to be a DataLoader object
    model.eval()

    preds = []
    pred_label = []
    for batch in tqdm(val_loader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            X, y, attention_mask = batch
            output = model(X, attention_mask=attention_mask, labels=y)
            logits = output.logits

            labels = logits.argmax(dim=1)
            preds.extend(logits.cpu().tolist())
            pred_label.extend(labels.cpu().tolist())

    return preds, pred_label

    

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/prediction.py', type=str, help='path to local model')
    parser.add_argument('--dataset_name', default='', type=str, help='name of dataset to be used')
    parser.add_argument('--device', default='cpu', type=str, help='decive to train on')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str, help='tokenizer model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = set_args()
    # load model from local file
    model_path = args.model_path
    model_name = args.model_name
    device = args.device
    save_path = '/home/featurize/work/saved/data/'
    dataset_name = args.dataset_name  # the path to the dataset, the dataset is a .csv file stored in the ../data directory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = SentimentDataset(tokenizer, [dataset_name])
    pred_df = pd.DataFrame(dataset.data, columns=["review", "gpt-label"])
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    if os.path.exists(model_path) == True:
        model = torch.load(model_path, map_location='cpu').to(device)    
        preds, pred_label = predict(model, test_loader, 'cuda:0')

        # turn the results into a dataframe
        pred_df['label_0-pred'] = [row[0] for row in preds]
        pred_df['label_1-pred'] = [row[1] for row in preds]
        pred_df['pred-label'] = pred_label

        pred_df.to_csv(save_path + 'prediction.csv', index=False)
