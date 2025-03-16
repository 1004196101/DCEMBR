#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2024/11/1 17:20
# @Desc  :
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_set import DataSet
from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN
from HDG import HDG

class DCEMBR(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(DCEMBR, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.reg_weight = args.reg_weight
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix
        self.user_item_inter_set = dataset.user_item_inter_set
        self.test_users = list(dataset.test_interacts.keys())
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.bhv_embs = nn.Parameter(torch.eye(len(self.behaviors)))
        self.hgd=HDG(self.device,self.n_users + 1, self.n_items + 1, dataset.all_inter_matrix)
        self.behavior_weights = nn.Parameter(torch.ones(len(self.behaviors)))
        self.Graph_encoder = nn.ModuleDict({
            behavior: LightGCN(
                self.device,
                2,
                self.n_users + 1,
                self.n_items + 1,
                dataset.inter_matrix[i]
            )
            for i, behavior in enumerate(self.behaviors)  # 同时获取行为和索引
        })
        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self, pre_embeddings):
        total_embeddings = pre_embeddings
        accumulated_embeddings = torch.zeros_like(total_embeddings)
        all_embeddings = {}
        for i, behavior in enumerate(self.behaviors):
            layer_embeddings=total_embeddings
            layer_embeddings = self.Graph_encoder[behavior](layer_embeddings)
            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            total_embeddings = layer_embeddings + total_embeddings
            all_embeddings[behavior] = total_embeddings
            weight = torch.sigmoid(self.behavior_weights[i])
            accumulated_embeddings += weight * all_embeddings[behavior]
        return accumulated_embeddings,all_embeddings


    def forward(self, batch_data):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.hgd(all_embeddings)
        buy_embeddings,_ = self.gcn_propagate(all_embeddings)
        user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings, [self.n_users + 1, self.n_items + 1])
        pair_samples = batch_data[:, -1, :-1]
        mask = torch.any(pair_samples != 0, dim=-1)
        pair_samples = pair_samples[mask]
        bpr_loss = 0
        if pair_samples.shape[0] > 0:
            user_samples = pair_samples[:, 0].long()
            item_samples = pair_samples[:, 1:].long()
            u_gen_emb = user_buy_embedding[user_samples].unsqueeze(1)
            i_final = item_buy_embedding[item_samples]
            score_gen = torch.sum((u_gen_emb * i_final), dim=-1)
            bpr_scores =  score_gen
            p_scores, n_scores = torch.chunk(bpr_scores, 2, dim=-1)
            bpr_loss += self.bpr_loss(p_scores, n_scores)
        emb_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)
        loss = bpr_loss + self.reg_weight * emb_loss
        return loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.hgd(all_embeddings)
            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            _,buy_embeddings=self.gcn_propagate(all_embeddings)
            user_buy_embedding, item_buy_embedding = torch.split(buy_embeddings[self.behaviors[-1]], [self.n_users + 1, self.n_items + 1])
            self.storage_user_embeddings = torch.zeros(self.n_users + 1, self.embedding_size).to(self.device)
            test_users = [int(x) for x in self.test_users]
            tmp_emb_list = []
            for i in range(0, len(test_users), 100):
                tmp_users = test_users[i: i + 100]
                tmp_users = torch.LongTensor(tmp_users)
                tmp_embeddings = user_embedding[tmp_users].unsqueeze(1)
                tmp_emb_list.append(tmp_embeddings.squeeze())
            user_embedding = user_buy_embedding
            self.storage_user_embeddings = torch.cat((self.storage_user_embeddings, user_embedding), dim=-1)
            self.storage_item_embeddings = torch.cat((item_embedding,item_buy_embedding), dim=-1)
        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))
        return scores

