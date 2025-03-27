import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data import Rna8fByRnaFamilyDataset
from data import get_one_sample_for_predict
from model import Model

import argparse
import json
import os

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch_num', type=int, default=100)

    parser.add_argument('--num_embeddings', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=20)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=500)
    parser.add_argument('--padding_idx', type=int, default=0)

    parser.add_argument('--dataset_root_path', type=str, default='processing_1/final_np_dataset_by_seq_len_train_valid_test/')
    parser.add_argument('--sub_dataset_list', type=str, default='150:500')

    parser.add_argument('--result_path', type=str, default='results/exp_main')
    parser.add_argument('--device', type=str, default='cuda')

    # for exp cross
    parser.add_argument('--test_rna_family', type=str, default='16srrna')

    return parser.parse_args()

rna_family_names = [
    '16srrna', 
    '5srrna',
    'introngrp1', 
    'rnasep',
    'srp', 
    'telomerase', 
    'trna',
    'tmrna'
]

args = get_args()

batch_size = args.batch_size
lr = args.lr
epoch_num = args.epoch_num

num_embeddings = args.num_embeddings
d_model = args.d_model
h = args.h
d_ff = args.d_ff
N = args.N
seq_len = args.seq_len
padding_idx = args.padding_idx

dataset_root_path = args.dataset_root_path
sub_dataset_list = args.sub_dataset_list.split(':')

result_path = args.result_path
device = args.device

test_rna_family = args.test_rna_family
train_rna_family_list = [i for i in rna_family_names if i != test_rna_family]
print(f'Train: {train_rna_family_list}')
print(f'Test: {test_rna_family}')

train_dataset = Rna8fByRnaFamilyDataset(
    dataset_root_path=dataset_root_path, 
    sub_dataset_list=sub_dataset_list, 
    dataset_type='train', 
    max_seq_len=seq_len,
    rna_family_name_list=train_rna_family_list)
# valid_dataset = Rna8fSubsetsDataset(
#     dataset_root_path=dataset_root_path, 
#     sub_dataset_list=sub_dataset_list, 
#     dataset_type='valid', 
#     max_seq_len=seq_len)
# test_dataset = Rna8fSubsetsDataset(
#     dataset_root_path=dataset_root_path, 
#     sub_dataset_list=sub_dataset_list, 
#     dataset_type='test', 
#     max_seq_len=seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)

# valid_sample_path_list = (
#     os.path.join(dataset_root_path, sub_dataset, 'valid') for sub_dataset in sub_dataset_list)
test_sample_path_list = []
for sub_dataset in sub_dataset_list:
    this_test_path = os.path.join(dataset_root_path, sub_dataset, test_rna_family)
    if not os.path.exists(this_test_path):
        continue
    test_sample_path_list.append(this_test_path)

assert len(test_sample_path_list) >= 1

print(f'Device: {device}')

model = Model(
    num_embeddings=num_embeddings, 
    padding_idx=padding_idx, 
    d_model=d_model, 
    h=h, 
    d_ff=d_ff, 
    N=N, 
    seq_len=seq_len).to(device)

loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

config_dict = {
    'batch_size': batch_size,
    'lr': lr,
    'epoch_num': epoch_num,
    'num_embeddings': num_embeddings,
    'd_model': d_model,
    'h': h,
    'd_ff': d_ff,
    'N': N,
    'seq_len': seq_len,
    'padding_idx': padding_idx,
    'dataset_root_path': dataset_root_path,
    'sub_dataset_list': sub_dataset_list,
    'result_path': result_path,
    'device': device,
    'test_rna_family': test_rna_family
}
with open(os.path.join(result_path, 'config.json'), 'w') as f:
    json.dump(config_dict, f, indent=4)

def train_valid_test():
    with open(os.path.join(result_path, 'valid_test_result.csv'), 'w') as f:
        f.write('epoch,type,metric,value\n')

    # best_valid_f1 = 0

    for epoch in range(epoch_num):
        print(f'Epoch: {epoch}')

        train_one_epoch()
        # valid_res = valid_test(valid_sample_path_list)
        test_res = valid_test(test_sample_path_list)
        
        # if valid_res['F1'] > best_valid_f1:
        #     print(f'Valid F1 increased: {best_valid_f1} -> {valid_res['F1']}, update model')
        #     best_valid_f1 = valid_res['F1']
        #     torch.save(model.state_dict(), os.path.join(result_path, f'model_state_dict.pt'))

        with open(os.path.join(result_path, 'valid_test_result.csv'), 'a') as f:

            # f.write(f"{epoch},valid,P,{valid_res['P']}\n")
            # f.write(f"{epoch},valid,R,{valid_res['R']}\n")
            # f.write(f"{epoch},valid,F1,{valid_res['F1']}\n")

            f.write(f"{epoch},test,P,{test_res['P']}\n")
            f.write(f"{epoch},test,R,{test_res['R']}\n")
            f.write(f"{epoch},test,F1,{test_res['F1']}\n")

        # no valid, just save the model in each epoch
        # and use the last model to test
        torch.save(model.state_dict(), os.path.join(result_path, f'model_state_dict.pt'))
        

def train_one_epoch():
    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = loss_fn(pred, y)

        # the loss function is not reduced, use a weighted sum
        # y shape is (minibatch, d1) (use class index instead of a C vector)
        diag_elements = (y == torch.arange(y.size(1)).to(device)) # (minibatch, d1)
        diag_num_per_batch = diag_elements.sum()
        not_diag_num_per_batch = y.size(0) * y.size(1) - diag_num_per_batch
        weight_for_diag_per_batch = not_diag_num_per_batch / diag_num_per_batch
        # if reduction is 'none', the shape of loss is also (minibatch, d1)
        loss[diag_elements] *= weight_for_diag_per_batch
        # then reduce the loss
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        print(f'Step {step + 1:>6d}, current loss {current_loss:>10.6f}')


def valid_test(sample_path_list):
    model.eval()
    
    sample_num = 0
    p_sum, r_sum, f1_sum = 0.0, 0.0, 0.0
    for sample_path in sample_path_list:
        for file_name in os.listdir(sample_path):
            if file_name.endswith('_x.npy'):
                with torch.no_grad():
                    main_name = file_name[:-6]
                    this_seq_len = int(main_name.split('_')[1])

                    x_npy = np.load(os.path.join(sample_path, file_name))
                    x = get_one_sample_for_predict(x_npy, seq_len).to(device)
                    predict = model(x).squeeze(0)[:this_seq_len, :this_seq_len]

                    y_npy = np.load(os.path.join(sample_path, main_name + '_y.npy'))
                    label = torch.zeros((this_seq_len, this_seq_len)).to(device)
                    if y_npy.size > 0:
                        label[y_npy[:, 0], y_npy[:, 1]] = 1.0
                    
                    tp = torch.sum(predict * label)
                    fp = torch.sum(predict * (1 - label))
                    fn = torch.sum((1 - predict) * label)
                    tn = torch.sum((1 - predict) * (1 - label))
                    p = tp / (tp + fp + 1e-10)
                    r = tp / (tp + fn + 1e-10)
                    f1 = 2 * p * r / (p + r + 1e-10)
                    p_sum += p.item()
                    r_sum += r.item()
                    f1_sum += f1.item()
                    sample_num += 1
    print(f'P {p_sum / sample_num:>10.6f}, R {r_sum / sample_num:>10.6f}, F1 {f1_sum / sample_num:>10.6f}')
    return {'P': p_sum / sample_num, 'R': r_sum / sample_num, 'F1': f1_sum / sample_num}


train_valid_test()
print('Done')