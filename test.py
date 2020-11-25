import argparse
import numpy as np
import torch
from utils import *
from model import MT
import pickle

def test(args, path=None):
    base_path = './data/' + args.dataset

    static_path = base_path+'/static.txt'
    args.ents, args.rels =  get_totabl_number(static_path)
    
    rel2val_path = base_path + '/rel2val.txt'
    r_val = get_r_val(rel2val_path)
    
    train_path, valid_path, test_path = base_path+'/train.txt', base_path+'/valid.txt', base_path+'/test.txt'
    train_data, valid_data, test_data = load_data(train_path), load_data(valid_path), load_data(test_path)
    total_data = np.concatenate((train_data, valid_data, test_data), axis=0)
    total_data = torch.from_numpy(total_data)
    test_data = torch.from_numpy(test_data)
    
    
    args.use_cuda = args.cuda >= 0 and torch.cuda.is_available()
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.use_cuda:
        torch.cuda.set_device(args.cuda)
        r_val = torch.from_numpy(r_val).cuda().float()
        
    model = MT(r_val, args)
    if args.use_cuda:
        model.cuda()
        total_data = total_data.cuda()
        

    test_s_r_history, test_r_o_history, test_s_o_history = base_path + '/test_s_r_history.txt', base_path + '/test_r_o_history.txt', base_path + '/test_s_o_history.txt'
    test_o_r_history, test_r_s_history, test_o_s_history = base_path + '/test_o_r_history.txt', base_path + '/test_r_s_history.txt', base_path + '/test_o_s_history.txt'

    test_s_r_o_history = load_history(test_s_r_history, test_r_o_history, test_s_o_history)
    test_o_r_s_history = load_history(test_o_r_history, test_r_s_history, test_o_s_history)
    

    config = '/_' + str(args.dim) + '_' + str(args.lr) + '_' + str(args.b) + '_' + str(args.dropout) + '_' + str(
        args.seq) + '_' + str(args.epochs) + '_' + str(args.model)
    model_state_file = 'models/'+args.dataset + config + "_MT.path"
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    
    print("Using best epoch: {}".format(checkpoint['epoch']), file=path)
    
    model.eval()
    total_loss = 0
    total_ranks = np.array([])
    total_ranks_s = np.array([])
    total_ranks_o = np.array([])

    for i in range(len(test_data)):
        batch_data = test_data[i]
        s_r_o_hist = [test_s_r_o_history[0][i], test_s_r_o_history[1][i], test_s_r_o_history[2][i]]
        o_r_s_hist = [test_o_r_s_history[0][i], test_o_r_s_history[1][i], test_o_r_s_history[2][i]]
        batch_data = batch_data.cuda()
        with torch.no_grad():
            if args.filter == 1:
                rank_s, rank_o, loss = model.evaluate_filter(batch_data, s_r_o_hist, o_r_s_hist, total_data)
            else:
                rank_s, rank_o, loss = model.evaluate(batch_data, s_r_o_hist, o_r_s_hist)
                
            total_ranks_s = np.concatenate((total_ranks_s, rank_s))
            total_ranks_o = np.concatenate((total_ranks_o, rank_o))
            ranks = np.concatenate((rank_s, rank_o))
            total_ranks = np.concatenate((total_ranks, ranks))
            total_loss += loss.item()

    mrr, mr = np.mean(1.0 / total_ranks), np.mean(total_ranks)
    hits, hits_s, hits_o = [], [], []
    for hit in [1, 3, 10]:
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("valid Hits (filtered) @ {}: {:.6f}".format(hit, avg_count), file=path)
    print("valid MRR (filtered): {:.6f}".format(mrr), file=path)
    print("valid MR (filtered): {:.6f}".format(mr), file=path)
    print("valid Loss: {:.6f}".format(total_loss / (len(valid_data))), file=path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MT')
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("-dim", type=int, default=256, help="number of hidden units")
    parser.add_argument("-cuda", type=int, default=0, help="gpu")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-dataset", type=str, default='ICEWS', help="dataset to use")
    parser.add_argument("-grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("-epochs", type=int, default=20, help="maximum epochs")
    parser.add_argument("-seq", type=int, default=10)
    parser.add_argument("-filter", type=int, default=1)
    parser.add_argument("-b", type=int, default=1024)
    parser.add_argument("-model", type=int, default=0)

    args = parser.parse_args()

    config = '/_' + str(args.dim) + '_' + str(args.lr) + '_' + str(args.b) + '_' + str(args.dropout) + '_' + str(
        args.seq) + '_' + str(args.epochs) + '_' + str(args.model)
    file_path = './result/' + str(args.dataset) + config +'_test.out'
    if args.filter != 1:
        file_path = file_path + '_nofilter'

    path = open(file_path, 'a', encoding='utf-8')
    print(config + str(args.filter), file=path)
    test(args, path)
    path.close()
