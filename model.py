import numpy as np
import torch.nn as nn
import torch
import sys
import torch.nn.functional as F
from Aggregator import Aggregator
from utils import *

class MT(nn.Module):
    def __init__(self, r_val, args):
        super(MT,self).__init__()
        self.args = args
        self.ents = args.ents
        self.rels = args.rels
        self.h_dim = args.dim
        self.r_val = r_val
        self.latest_time = 0
        
        self.rel_embeds = nn.Parameter(torch.Tensor(self.rels*2, self.h_dim))
        nn.init.xavier_uniform_(self.rel_embeds,gain=nn.init.calculate_gain('relu'))
        self.ent_embeds = nn.Parameter(torch.Tensor(self.ents, self.h_dim))
        nn.init.xavier_uniform_(self.ent_embeds, gain=nn.init.calculate_gain('relu'))
        
        if args.model == 0:
            self.encoder = nn.GRU(3*self.h_dim, self.h_dim, batch_first=True)
            self.encoder_o = self.encoder#nn.GRU(3*self.h_dim, self.h_dim, batch_first=True)
        elif args.model == 1:
            self.encoder = nn.LSTM(3 * self.h_dim, self.h_dim, batch_first=True)
            self.encoder_o = self.encoder#nn.LSTM(3 * self.h_dim, self.h_dim, batch_first=True)
        elif args.model == 2:
            self.encoder = nn.RNN(3 * self.h_dim, self.h_dim, batch_first=True)
            self.encoder_o = self.encoder#nn.RNN(3 * self.h_dim, self.h_dim, batch_first=True)
        else:
            self.encoder = nn.GRU(3 * self.h_dim, self.h_dim, batch_first=True)
            self.encoder_o = self.encoder#nn.GRU(3 * self.h_dim, self.h_dim, batch_first=True)
        
        self.aggregator = Aggregator(self.h_dim, args.dropout, args.seq)
        self.aggregator_o = self.aggregator
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(3*self.h_dim, self.ents)
        self.linear_o = self.linear#nn.Linear(3*self.h_dim, self.ents)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_o = self.criterion#nn.CrossEntropyLoss()
        
    def forward(self, triplets, s_r_o_hist, o_r_s_hist):
        s = triplets[:,0]
        r = triplets[:,1]
        o = triplets[:,2]
        
        hist_len = torch.LongTensor(list(map(len,s_r_o_hist[0]))).cuda()
        hist_len_o = torch.LongTensor(list(map(len,o_r_s_hist[0]))).cuda()
        length, idx = hist_len.sort(0,descending=True)
        length_o, idx_o = hist_len_o.sort(0,descending=True)
        
        packed_input_s = self.aggregator(s_r_o_hist, self.r_val, s, r, self.ent_embeds, self.rel_embeds[:self.rels])
        packed_input_o = self.aggregator_o(o_r_s_hist, self.r_val, o, r, self.ent_embeds,self.rel_embeds[self.rels:])
        
        tt, h_s = self.encoder(packed_input_s)
        tt, h_o = self.encoder_o(packed_input_o)
        h_s = h_s.squeeze()
        h_o = h_o.squeeze()
        h_s = torch.cat((h_s, torch.zeros(len(s) - len(h_s), self.h_dim).cuda()), dim=0)
        h_o = torch.cat((h_o, torch.zeros(len(o) - len(h_o), self.h_dim).cuda()), dim=0)
        pred_o = self.linear(self.dropout(torch.cat((self.ent_embeds[s[idx]], h_s, self.rel_embeds[:self.rels][r[idx]]), dim=1)))
        pred_s = self.linear_o(self.dropout(torch.cat((self.ent_embeds[o[idx_o]], h_o, self.rel_embeds[self.rels:][r[idx_o]]), dim=1)))
        loss_s = self.criterion(pred_o, o[idx])
        loss_o = self.criterion_o(pred_s, s[idx_o])
        loss = loss_s+loss_o
        
        return loss, pred_o, pred_s, idx
    
    def get_loss(self, triplets,  s_r_o_hist, o_r_s_hist):
        loss, _, _, _ = self.forward(triplets, s_r_o_hist, o_r_s_hist)
        return loss
    
    def predict(self, batch_data, s_r_o_hist, o_r_s_hist):
        s = batch_data[0]
        r = batch_data[1]
        o = batch_data[2]
        
        if len(s_r_o_hist[0]) == 0:
            h = torch.zeros(self.h_dim).cuda()
        else:
            inp = self.aggregator.predict(s_r_o_hist, self.r_val, s, r, self.ent_embeds, self.rel_embeds[:self.rels])
            tt, h = self.encoder(inp.view(1, len(s_r_o_hist[0]), 3 * self.h_dim))
            h = h.squeeze()
        if len(o_r_s_hist[0]) == 0:
            h_o = torch.zeros(self.h_dim).cuda()
        else:
            inp_o = self.aggregator_o.predict(o_r_s_hist, self.r_val, o, r, self.ent_embeds, self.rel_embeds[self.rels:])
            tt, h_o = self.encoder_o(inp_o.view(1,len(o_r_s_hist[0]),3*self.h_dim))
            h_o = h_o.squeeze()
            
        pred_o = self.linear(torch.cat((self.ent_embeds[s],h,self.rel_embeds[:self.rels][r]),dim=0))
        pred_s = self.linear_o(torch.cat((self.ent_embeds[o],h_o,self.rel_embeds[self.rels:][r]),dim=0))

        loss_s = self.criterion(pred_o.view(1,-1),o.view(-1))
        loss_o = self.criterion_o(pred_s.view(1,-1),s.view(-1))
        loss = loss_s + loss_o
                
        return loss, pred_o, pred_s
    
    def evaluate(self, batch_data, s_r_o_hist, o_r_s_hist):
        s, r, o = batch_data[0], batch_data[1], batch_data[2]

        loss, pred_o, pred_s = self.predict(batch_data, s_r_o_hist, o_r_s_hist)
        o_label = o
        s_label = s
        ob_pred_comp1 = (pred_o > pred_o[o_label]).data.cpu().numpy()
        ob_pred_comp2 = (pred_o == pred_o[o_label]).data.cpu().numpy()
        rank_s = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1

        sub_pred_comp1 = (pred_s > pred_s[s_label]).data.cpu().numpy()
        sub_pred_comp2 = (pred_s == pred_s[s_label]).data.cpu().numpy()
        rank_o = np.sum(sub_pred_comp1) + ((np.sum(sub_pred_comp2) - 1.0) / 2) + 1

        return np.array([rank_s]), np.array([rank_o]), loss
        
    
    def evaluate_filter(self, batch_data, s_r_o_hist, o_r_s_hist, total_data):
        s = batch_data[0]
        r = batch_data[1]
        o = batch_data[2]
        loss, pred_o, pred_s= self.predict(batch_data, s_r_o_hist, o_r_s_hist)
        
        label = o
        pred_o = F.sigmoid(pred_o)
        ground = pred_o[o].clone()
        s_id = torch.nonzero(total_data[:, 0] == s).view(-1)
        idx = torch.nonzero(total_data[s_id, 1] == r).view(-1)
        idx = s_id[idx]
        idx = total_data[idx, 2]
        pred_o[idx] = 0
        pred_o[label] = ground
        pred_comp1 = (pred_o > ground).data.cpu().numpy()
        pred_comp2 = (pred_o == ground).data.cpu().numpy()
        rank_s = np.sum(pred_comp1) + ((np.sum(pred_comp2) - 1.0) / 2) + 1
        
        label = s
        pred_s = F.sigmoid(pred_s)
        ground = pred_s[s].clone()
        o_id = torch.nonzero(total_data[:, 2] == o).view(-1)
        idx = torch.nonzero(total_data[o_id, 1] == r).view(-1)
        idx = o_id[idx]
        idx = total_data[idx, 0]
        pred_s[idx] = 0
        pred_s[label] = ground
        pred_comp1 = (pred_s > ground).data.cpu().numpy()
        pred_comp2 = (pred_s == ground).data.cpu().numpy()
        rank_o = np.sum(pred_comp1) + ((np.sum(pred_comp2) - 1.0) / 2) + 1
        
        return np.array([rank_s]), np.array([rank_o]), loss
