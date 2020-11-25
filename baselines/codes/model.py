#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        if self.model_name=='RESCAL':
            self.embedding_range = nn.Parameter(
                torch.Tensor([0.31]),
                requires_grad=False
            )
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
                requires_grad=False
            )

        
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if(self.model_name=='RESCAL'):
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim,self.relation_dim))
            self.entity_embedding = nn.Parameter(torch.zeros(nentity,1,self.entity_dim))
 
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        
        
    def forward(self, sample, mode='single'):

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            if(self.model_name=='RESCAL'):
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size,1,-1)
            else:
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if(self.model_name=='RESCAL'):
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, 1,-1)
            else:
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'DistMult': self.DistMult,
            'RESCAL':self.RESCAL,
            }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        #score = nn.Softmax(dim=1)(score)
        return score

    def RESCAL(self,head, relation, tail, mode):
        tail1 = tail.permute(0, 1, 3, 2)
        if mode == 'head-batch':
            k = torch.matmul(relation, tail1)
            score = torch.matmul(head, k)
        else:
            k = torch.matmul(head, relation)
            score = torch.matmul(k, tail1)
        score = score.squeeze(dim=2)
        score = score.squeeze(dim=2)
        #score = nn.Softmax(dim=1)(score)
        return score
    
  
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):


        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        positive_score = model(positive_sample)

        pos_loss = torch.sigmoid(positive_score)
        pos_loss = -torch.log(pos_loss+0.000000001)
        pos_loss = pos_loss.squeeze(dim=1)
        pos_loss = (subsampling_weight * pos_loss).sum() / subsampling_weight.sum()

        neg_loss = torch.sigmoid(negative_score)
        neg_loss = -(args.uncertainty*torch.log(neg_loss+0.000000001)+(1-args.uncertainty)*torch.log(1-neg_loss+0.000000001)).mean(dim = 1)
        neg_loss = (subsampling_weight * neg_loss).sum() / subsampling_weight.sum()

        loss = (pos_loss + neg_loss) / 2

        regularization = 0.000005 * (model.entity_embedding.norm(p = 3)**3 + model.relation_embedding.norm(p = 3).norm(p = 3)**3)
        
        loss = loss + regularization
        
        loss.backward()

        optimizer.step()

        log = {
            
            'loss': loss.item()
        }

        return log,loss.item()
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        
        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=1,
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    # explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
        
            
