import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils import *

class Aggregator(nn.Module):
    def __init__(self, h_dim, dropout, seq_len=10):
        super(Aggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.gcn_layer = nn.Linear(h_dim*2, h_dim*2)
            
    def forward(self, hist, r_val, ent, rel, ent_embeds, rel_embeds):
        len_non_zero, s_tem, embeds_stack, length, embeds_split, rel_weights_tensor = get_sorted_embeds(hist[0], hist[1], hist[2], r_val, ent, rel, ent_embeds, rel_embeds)
        curr, rows, cols = 0, [], []
        for i, leng in enumerate(length):
            rows.extend([int(i)] * leng)  # 在列表末尾一次性追加另一个序列中的多个值
            cols.extend(list(range(curr, curr + leng)))
            curr += leng
        
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        idxes = torch.stack([rows, cols], dim=0).cuda()
        
        mask_tensor = torch.sparse.FloatTensor(idxes,rel_weights_tensor)  # 返回一个全为1的张量,形状由可变参数sizes定义
        mask_tensor = mask_tensor.cuda()
        embeds_sum = torch.sparse.mm(mask_tensor, embeds_stack)
        embeds_mean = embeds_sum / torch.Tensor(length).cuda().view(-1, 1)
        
        embeds_mean = self.gcn_layer(embeds_mean)
        embeds_mean = torch.relu(embeds_mean)
        embeds_split = torch.split(embeds_mean, len_non_zero.tolist())
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 3 * self.h_dim).cuda()
        
        for i, embeds in enumerate(embeds_split):
            embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1)), dim=1)
        
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor, len_non_zero, batch_first=True)
        
        return packed_input
    
    def predict(self, hist, r_val, s, r, ent_embeds, rel_embeds):
        inp = torch.zeros(len(hist[0]), 3 * self.h_dim).cuda()
        for i, rel in enumerate(hist[0]):
            r_o_hist, s_o_hist = hist[1][i], hist[2][i]
            length_r_o, length_s_o = [], []
            flat_r_o, flat_s_o = [], []
            for objs in r_o_hist:
                length_r_o.append(len(objs))
                for obj in objs:
                    flat_r_o.append(obj)
            tem_obj_embeds = ent_embeds[torch.LongTensor(flat_r_o).cuda()]
            tem_rel_embeds = rel_embeds[rel]
            
            for vals in s_o_hist:
                length_s_o.append(len(vals))
                for val in vals:
                    flat_s_o.append(val)
            curr, rows, cols = 0, [], []
            for num, leng in enumerate(length_r_o):
                rows.extend([int(num)] * leng)
                cols.extend(list(range(curr, curr + leng)))
                curr += leng
            rows, cols = torch.LongTensor(rows), torch.LongTensor(cols)
            idxes = torch.stack([rows, cols], dim=0).cuda()
            rel_val_tensor = r_val[r].repeat(len(rel), 1)
            
            rel_val = r_val[torch.LongTensor(rel).cuda()].reshape(-1,1)
            
            rel_one_tensor = torch.ones(len(rel), dtype=torch.float).cuda().reshape(-1,1)
            rel_weight_tensor = rel_one_tensor.sub(torch.abs(rel_val_tensor.sub(rel_val)))
            
            one_tensor = torch.ones(len(flat_r_o), dtype=torch.float).cuda()
            obj_rel_val = rel_val.data.cpu().numpy()
            obj_val_array = []

            for num, leng in enumerate(length_r_o):
                for index in range(leng):
                    obj_val_array.append(obj_rel_val[num])
            obj_val_tensor = torch.FloatTensor(obj_val_array).cuda()
            os_val = torch.FloatTensor(flat_s_o).cuda()
            obj_abs_vector = torch.abs(obj_val_tensor.sub(os_val))
            obj_weights_tensor = one_tensor.sub(obj_abs_vector)  # .reshape((-1, 1))
            obj_mask_tensor = torch.sparse.FloatTensor(idxes, obj_weights_tensor)  # torch.ones(len(rows)))
            obj_mask_tensor = obj_mask_tensor.cuda()
            tem_obj_tensor = torch.sparse.mm(obj_mask_tensor, tem_obj_embeds)
            tem_obj_tensor = tem_obj_tensor / torch.Tensor(length_r_o).cuda().view(-1, 1)
            tem_rel_obj = torch.cat((tem_rel_embeds, tem_obj_tensor), dim=1)
            
            tem_rel_obj = torch.mul(tem_rel_obj, rel_weight_tensor)
            embeds_val = torch.mean(tem_rel_obj, dim=0)
            
            embeds_val = torch.relu(self.gcn_layer(embeds_val))
            inp[i] = torch.cat((embeds_val, ent_embeds[s]), dim=0)
        
        return inp
