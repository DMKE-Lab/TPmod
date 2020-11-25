import numpy as np
import os
import sys
import torch
import pickle
from collections import defaultdict

def get_totabl_number(fileName):
    with open(fileName, 'r') as fr:
        line = fr.readline()
        line_split = line.split()
        return list(map(int, line_split))[:2]
        
def get_r_val(fileName):
    val = []
    with open(fileName, 'r') as fr:
        for line in fr:
            line_split = line.split()
            val.append(float(line_split[1]))
    return np.asarray(val)

def load_data(fileName):
    with open(fileName, 'r') as fr:
        quadrupleList = []
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
    return np.asarray(quadrupleList)

def load_history(file1, file2, file3):
    with open(file1, 'rb') as f:
        train_history1 = pickle.load(f)
    with open(file2, 'rb') as f:
        train_history2 = pickle.load(f)
    with open(file3, 'rb') as f:
        train_history3 = pickle.load(f)
    return [train_history1, train_history2, train_history3]

def make_batch(train_data, train_s_r_o_history, train_o_r_s_history, size):
    for i in range(0, len(train_data), size):
        yield  train_data[i:i+size], [train_s_r_o_history[0][i:i+size], train_s_r_o_history[1][i:i+size], train_s_r_o_history[2][i:i+size]] , \
               [train_o_r_s_history[0][i:i+size], train_o_r_s_history[1][i:i+size], train_o_r_s_history[2][i:i+size]]
        
def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def get_sorted_embeds(s_r_hist, r_o_hist, s_o_hist, r_val, s, r, ent_embeds, rel_embeds):
    hist_len = torch.LongTensor(list(map(len,s_r_hist))).cuda()
    length, idx = hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(length))
    len_non_zero = length[:num_non_zero]
    
    s_tem = s[idx]
    hist_sorted = []
    r_o_sroted= []
    s_o_sorted = []

    for id in idx:
        hist_sorted.append(s_r_hist[id.item()])
        r_o_sroted.append(r_o_hist[id.item()])
        s_o_sorted.append(s_o_hist[id.item()])
    hist_sorted = hist_sorted[:num_non_zero]
    s_o_sorted = s_o_sorted[:num_non_zero]
    r_o_sroted = r_o_sroted[:num_non_zero]
    
    flat = []
    length = []
    rel_index = []
    i = 0
    for hist in hist_sorted:
        for rels in hist:
            length.append(len(rels))
            for rel in rels:
                flat.append(rel)
                rel_index.append(i)
        i+=1
    
    flat_r_o = []
    flat_s_o = []
    length_r_o = []
    length_s_o = []
    for hists in  r_o_sroted:
        for hist in hists:
            for objs in hist:
                length_r_o.append(len(objs))
                for obj in objs:
                    flat_r_o.append(obj)
    for hists in s_o_sorted:
        for hist in hists:
            for vals in hist:
                length_s_o.append(len(vals))
                for val in vals:
                    flat_s_o.append(val)

    obj_embeds = ent_embeds[torch.LongTensor(flat_r_o).cuda()]
    rel_embeds = rel_embeds[torch.LongTensor(flat).cuda()]
    
    curr = 0
    rows = []
    cols = []
    for i, leng in enumerate(length_r_o):
        rows.extend([i] * leng)
        cols.extend(list(range(curr, curr + leng)))
        curr += leng
    rows = torch.LongTensor(rows)
    cols = torch.LongTensor(cols)
    idxes = torch.stack([rows, cols], dim=0).cuda()
    
    with torch.no_grad():
        rel_one_tensor = torch.ones(len(flat), dtype=torch.float).cuda()
        given_r_val = r_val[r].data.cpu().numpy()
        rel_val_array = []
        j = 0
        for i, leng in enumerate(length):
            for index in range(leng):
                rel_val_array.append(given_r_val[rel_index[j]])
                j += 1
        his_rel_val = r_val[torch.LongTensor(flat).cuda()]
        rel_val_tensor = torch.FloatTensor(rel_val_array).cuda()
        rel_abs_vector = torch.abs(rel_val_tensor.sub(his_rel_val))
        rel_weights_tensor = rel_one_tensor.sub(rel_abs_vector)
        
        his_rel_val = his_rel_val.data.cpu().numpy()
        his_rel_val_array = []
        for i, leng in enumerate(length_r_o):
            for index in range(leng):
                his_rel_val_array.append(his_rel_val[i])
        obj_one_tensor = torch.ones(len(flat_r_o), dtype=torch.float).cuda()
        obj_val_tensor = torch.FloatTensor(his_rel_val_array).cuda()
        os_val = torch.FloatTensor(flat_s_o).cuda()
        obj_abs_vector = torch.abs(obj_val_tensor.sub(os_val))
        obj_weights_tensor = obj_one_tensor.sub(obj_abs_vector)#.reshape((-1, 1))
        obj_mask_tensor = torch.sparse.FloatTensor(idxes, obj_weights_tensor)
        obj_mask_tensor = obj_mask_tensor.cuda()
        
    o_embeds = torch.sparse.mm(obj_mask_tensor, obj_embeds)
    o_embeds = o_embeds / torch.Tensor(length_r_o).cuda().view(-1, 1)
    
    embeds_val = torch.cat((rel_embeds,o_embeds), dim=1)
    embeds_split = torch.split(embeds_val, length)
    
    return len_non_zero,s_tem,embeds_val,length,embeds_split,rel_weights_tensor
