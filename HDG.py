import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.nn.init import xavier_uniform_
from recbole.utils import InputType
from spmm import SpecialSpmm, CHUNK_SIZE_FOR_SPMM

class HDG(nn.Module):
    input_type = InputType.PAIRWISE

    def __init__(self, device,n_users, n_items, interaction_matrix,prune_thresholds=[0.1,0.5]):
        super(HDG, self).__init__()

        # load base para
        self.n_layers = 2
        self.reg_weight = 1e-5
        self.device = device

        self.interaction_matrix = interaction_matrix
        self.n_users = n_users
        self.n_items = n_items

        self.adj_matrix = self._get_a_adj_matrix().to(self.device)
        self.edge_gate = nn.Linear(1, 1)
        xavier_uniform_(self.edge_gate.weight.data, gain=1.414)
        if self.edge_gate.bias is not None:
            self.edge_gate.bias.data.fill_(0.0)
        self.prune_thresholds = [0.05 for _ in range(self.n_layers)]
        self.spmm ="spmm"
        self.special_spmm = SpecialSpmm() if self.spmm == 'spmm' else torch.sparse.mm
        self.pool_multi = 10
        self.for_learning_adj()
        self.restore_user_e = None
        self.restore_item_e = None
        self.apply(self._init_weights)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def for_learning_adj(self):
        self.adj_matrix = self.adj_matrix.coalesce()
        self.adj_indices = self.adj_matrix.indices()
        self.adj_shape = self.adj_matrix.shape
        self.adj = self.adj_matrix
        inter_data = torch.FloatTensor(self.interaction_matrix.data).to(self.device)
        inter_user = torch.LongTensor(self.interaction_matrix.row).to(self.device)
        inter_item = torch.LongTensor(self.interaction_matrix.col).to(self.device)
        inter_mask = torch.stack([inter_user, inter_item], dim=0)
        self.inter_spTensor = torch.sparse.FloatTensor(inter_mask, inter_data, self.interaction_matrix.shape).coalesce()
        self.inter_spTensor_t = self.inter_spTensor.t().coalesce()
        self.inter_indices = self.inter_spTensor.indices()
        self.inter_shape = self.inter_spTensor.shape

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def _get_a_adj_matrix(self):
        A = sp.dok_matrix((self.n_users  + self.n_items , self.n_users + self.n_items), dtype=float)
        inter_matrix = self.interaction_matrix
        inter_matrix_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_matrix.row, inter_matrix.col + self.n_users), [1] * inter_matrix.nnz))
        data_dict.update(
            dict(zip(zip(inter_matrix_t.row + self.n_users, inter_matrix_t.col), [1] * inter_matrix_t.nnz)))
        A._update(data_dict)
        sum_list = (A > 0).sum(axis=1)
        diag = np.array(sum_list.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        A_adj = D * A * D
        A_adj = sp.coo_matrix(A_adj)
        row = A_adj.row
        col = A_adj.col
        index = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_adj.data)
        A_sparse = torch.sparse.FloatTensor(index, data, torch.Size(A_adj.shape))
        return A_sparse
    def sp_cos_sim(self, a, b, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        L = self.inter_indices.shape[1]
        sims = torch.zeros(L, dtype=a.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            batch_indices = self.inter_indices[:, idx:idx + CHUNK_SIZE]
            valid_mask = (batch_indices[0, :] < a_norm.size(0)) & (batch_indices[1, :] < b_norm.size(0))
            batch_indices = batch_indices[:, valid_mask]
            if batch_indices.size(1) == 0:
                continue

            a_batch = torch.index_select(a_norm, 0, batch_indices[0, :])
            b_batch = torch.index_select(b_norm, 0, batch_indices[1, :])

            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + batch_indices.size(1)] = dot_prods

        return torch.sparse_coo_tensor(self.inter_indices, sims, size=self.interaction_matrix.shape,
                                       dtype=sims.dtype).coalesce()

    def get_sim_mat(self,all_embeddings):
        total_size = all_embeddings.size(0)
        user_size = min(self.n_users + 1, total_size)
        item_size = total_size - user_size  # 确保不会超出 total_size
        user_embedding, item_embedding = torch.split(all_embeddings, [user_size, item_size])
        user_feature = user_embedding.to(self.device)
        item_feature = item_embedding.to(self.device)
        sim_inter = self.sp_cos_sim(user_feature, item_feature)
        return sim_inter

    def inter2adj(self, inter):
        inter_t = inter.t().coalesce()
        data = inter.values()
        data_t = inter_t.values()
        adj_data = torch.cat([data, data_t], dim=0)
        adj = torch.sparse.FloatTensor(self.adj_indices, adj_data, self.adj_shape).to(self.device).coalesce()
        return adj
    def get_sim_adj(self, all_embeddings, layer_idx):
        sim_mat = self.get_sim_mat(all_embeddings)
        sim_adj = self.inter2adj(sim_mat)

        pruning = self.prune_thresholds[layer_idx]
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)
        gate_input = sim_value.unsqueeze(1)
        gate_weights = torch.sigmoid(self.edge_gate(gate_input)).squeeze(1)

        pruned_sim_value = sim_value * gate_weights
        pruned_sim_value = torch.where(pruned_sim_value < pruning, torch.zeros_like(pruned_sim_value), pruned_sim_value)
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, self.adj_shape).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value,
                                                  self.adj_shape).to(self.device).coalesce()

        return normal_sim_adj

    def forward(self, in_embs,pruning=0.0):
        all_embeddings = in_embs
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            self.adj = self.get_sim_adj(all_embeddings, layer_idx)
            all_embeddings = self.special_spmm(self.adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings


