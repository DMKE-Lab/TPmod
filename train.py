import argparse
import sys
import numpy as np
import time
import torch
from utils import *
import os
import pickle
from model import MT
from sklearn.utils import shuffle

def train_epoch(model, train_data, train_s_r_o_history, train_o_r_s_history, optimizer, args, path=None):
    model.train()
    for batch_data, s_r_o_hist, o_r_s_hist in make_batch(train_data, train_s_r_o_history, train_o_r_s_history, args.b):
        optimizer.zero_grad
        batch_data = torch.from_numpy(batch_data)
        if args.use_cuda:
            batch_data = batch_data.cuda()
        loss = model.get_loss(batch_data, s_r_o_hist, o_r_s_hist)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        loss += loss.item()

def valid_epoch(model, valid_data, valid_s_r_o_history, valid_o_r_s_history, total_data, args, path=None):
    model.eval()
    total_loss = 0
    total_ranks = np.array([])
    total_ranks_s = np.array([])
    total_ranks_o = np.array([])
    model.latest_time = valid_data[0][3]
    
    for i in range(len(valid_data)):
        batch_data = valid_data[i]
        s_r_o_hist = [valid_s_r_o_history[0][i], valid_s_r_o_history[1][i], valid_s_r_o_history[2][i]]
        o_r_s_hist = [valid_o_r_s_history[0][i], valid_o_r_s_history[1][i], valid_o_r_s_history[2][i]]
        batch_data = batch_data.cuda()
        with torch.no_grad():
            rank_s, rank_o, loss = model.evaluate_filter(batch_data, s_r_o_hist, o_r_s_hist, total_data)
            total_ranks_s = np.concatenate((total_ranks_s, rank_s))
            total_ranks_o = np.concatenate((total_ranks_o, rank_o))
            ranks = np.concatenate((rank_s, rank_o))
            total_ranks = np.concatenate((total_ranks, ranks))
            total_loss += loss.item()
    
    best_mrr = 0
    mrr, mr = np.mean(1.0 / total_ranks), np.mean(total_ranks)
    hits, hits_s, hits_o = [], [], []
    for hit in [1, 3, 10]:
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("valid Hits (filtered) @ {}: {:.6f}".format(hit, avg_count), file=path)
    print("valid MRR (filtered): {:.6f}".format(mrr), file=path)
    print("valid MR (filtered): {:.6f}".format(mr), file=path)
    print("valid Loss: {:.6f}".format(total_loss / (len(valid_data))), file=path)
    
    os.makedirs('models', exist_ok=True)
    config = '/_'  + str(args.dim) + '_' + str(args.lr) + '_' + str(args.b) + '_' + str(args.dropout) + '_' + str(args.seq) + '_' + str(args.epochs)  + '_' + str(args.model)
    model_state_file = 'models/'+args.dataset + config + "_MT.path"
    if mrr >= best_mrr:
        best_mrr = mrr
        torch.save(
            {'state_dict': model.state_dict(), 'epoch': args.epoch, 'latest_time': model.latest_time, 'best_mrr': best_mrr}, model_state_file)

def train(args):
   
    base_path = './data/' + args.dataset

    static_path = base_path+'/static.txt'
    args.ents, args.rels = get_totabl_number(static_path)
    
    rel2val_path = base_path + '/rel2val.txt'
    r_val = get_r_val(rel2val_path)
    
    train_path, valid_path, test_path = base_path+'/train.txt', base_path+'/valid.txt', base_path+'/test.txt'
    train_data, valid_data, test_data = load_data(train_path), load_data(valid_path), load_data(test_path)
    total_data = np.concatenate((train_data, valid_data, test_data), axis=0)

    args.use_cuda = args.cuda >= 0 and torch.cuda.is_available()
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.use_cuda:
        torch.cuda.set_device(args.cuda)
        r_val = torch.from_numpy(r_val).cuda().float()
    
    train_s_r_history, train_r_o_history, train_s_o_history = base_path + '/train_s_r_history.txt', base_path + '/train_r_o_history.txt', base_path + '/train_s_o_history.txt'
    train_o_r_history, train_r_s_history, train_o_s_history = base_path + '/train_o_r_history.txt', base_path + '/train_r_s_history.txt', base_path + '/train_o_s_history.txt'
    valid_s_r_history, valid_r_o_history, valid_s_o_history = base_path + '/valid_s_r_history.txt', base_path + '/valid_r_o_history.txt', base_path + '/valid_s_o_history.txt'
    valid_o_r_history, valid_r_s_history, valid_o_s_history = base_path + '/valid_o_r_history.txt', base_path + '/valid_r_s_history.txt', base_path + '/valid_o_s_history.txt'
    
    train_s_r_o_history = load_history(train_s_r_history, train_r_o_history, train_s_o_history)
    train_o_r_s_history = load_history(train_o_r_history, train_r_s_history, train_o_s_history)
    valid_s_r_o_history = load_history(valid_s_r_history, valid_r_o_history, valid_s_o_history)
    valid_o_r_s_history = load_history(valid_o_r_history, valid_r_s_history, valid_o_s_history)
    
    model = MT(r_val, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=0.0001)
    
    total_data = torch.from_numpy(total_data)
    valid_data = torch.from_numpy(valid_data)
    
    if args.use_cuda:
        model.cuda()
        total_data = total_data.cuda()
    
    args.epoch = 0
    while True:
        config = '/_' + str(args.dim) + '_' + str(args.lr) + '_' + str(args.b) + '_' + str(args.dropout) + '_' + str(
            args.seq) + '_' + str(args.epochs) + '_' + str(args.model)
        file_path = './result/' + str(args.dataset) + config + '_train.out'
        path = open(file_path, 'a', encoding='utf-8')
        print('start training', file=path)
        loss, args.epoch = 0, args.epoch+1
        if args.epoch == args.epochs:
            break
        train_epoch(model, train_data, train_s_r_o_history, train_o_r_s_history, optimizer, args, path)
        if args.epoch >= args.epochs-10:
            valid_epoch(model, valid_data, valid_s_r_o_history, valid_o_r_s_history, total_data, args, path)
        print("training done", file=path)
        path.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MT')
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("-dim", type=int, default=256, help="number of hidden units")
    parser.add_argument("-cuda", type=int, default=0, help="gpu")
    parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("-dataset", type=str, default='ICEWS-250', help="dataset to use")
    parser.add_argument("-epochs", type=int, default=50, help="maximum epochs")
    parser.add_argument("-seq", type=int, default=10)
    parser.add_argument("-b", type=int, default=1024)
    
    args = parser.parse_args()
    train(args)

