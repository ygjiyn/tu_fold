import torch
import numpy as np
from data import get_one_sample_for_predict
from model import Model
import json
import os

import argparse

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--sample_path_list', required=True)
    return parser.parse_args()

args = get_args()
results_folder_path = args.results_folder_path
save_path = args.save_path
sample_path_list = args.sample_path_list.split(':')

os.makedirs(save_path)

with open(os.path.join(results_folder_path, 'config.json')) as f:
    config_dict = json.load(f)

num_embeddings = config_dict['num_embeddings']
padding_idx = config_dict['padding_idx']
d_model = config_dict['d_model']
h = config_dict['h']
d_ff = config_dict['d_ff']
N = config_dict['N']
seq_len = config_dict['seq_len']
device = config_dict['device']

model = Model(
    num_embeddings=num_embeddings, 
    padding_idx=padding_idx, 
    d_model=d_model, 
    h=h, 
    d_ff=d_ff, 
    N=N, 
    seq_len=seq_len).to(device)

model.load_state_dict(torch.load(os.path.join(results_folder_path, f'model_state_dict.pt'), map_location=device))
model.eval()

for sample_path in sample_path_list:
    print(f'Processing: {sample_path}')
    for i, file_name in enumerate(os.listdir(sample_path)):
        if file_name.endswith('_x.npy'):
            with torch.no_grad():
                x_npy = np.load(os.path.join(sample_path, file_name))
                x = get_one_sample_for_predict(x_npy, seq_len).to(device)
                pred = model(x)
                pred = pred.squeeze(0).cpu().numpy()
                main_name = file_name[:-6]
                this_seq_len = int(main_name.split('_')[1])
                np.save(os.path.join(save_path, main_name + '.npy'), pred[:this_seq_len, :this_seq_len])
