import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv


class HeadlineEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim)

    def forward(self, x_headline, headline_lengths):
        """
        Args:
         - x_headline: [N, L_headline, H]
         - headline_lengths: [N]
        Returns:
         - h_headline: [N, H']
        """
        _, h_headline = self.rnn(pack_padded_sequence(x_headline, headline_lengths, batch_first=True,
                                                      enforce_sorted=False))  # -> _, [1, N, H']
        return h_headline.squeeze(dim=0)  # -> [N, H']


class BodyEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim)

    def forward(self, x_body, seq_lengths, num_seq_in_body, return_masked=False):
        """
        Args:
         - x_body: [N, S, L, H] (S=max_num_seq_in_body, L=max_seq_length)
         - seq_lengths: [N, S]
         - num_seq_in_body: [N]
        Returns:
         - h_body: [N, S, H']
        """
        # merge dimensions N and S for parallel processing
        x_body_masked = x_body[seq_lengths != 0]  # -> [sum(S_i), H]
        seq_lengths_masked = seq_lengths[seq_lengths != 0] # -> [sum(S_i)]

        _, h_body_masked = self.rnn(pack_padded_sequence(x_body_masked, seq_lengths_masked, batch_first=True,
                                                         enforce_sorted=False))  # -> [1, sum(S_i), H']
        h_body_masked = h_body_masked.squeeze(dim=0)  # -> [sum(S_i), H']

        # unmerge dimensions N and S
        h_body_grouped = h_body_masked.split(num_seq_in_body.tolist())  # -> tuple of N tensors of shape [S_i, H']
        h_body = pad_sequence(h_body_grouped, batch_first=True)  # -> [N, S, H']

        if return_masked:
            return h_body, h_body_masked
        return h_body


class SequenceEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.rnn = nn.GRU(in_features, out_features, bidirectional=True)

    def forward(self, h, seq_lengths):  # [N, L, H], [N]
        """
        Args:
         - h: [N, L, H] (L=max_seq_length)
         - seq_lengths: [N]
        Returns:
         - u: [N, L, H']
        """
        _u_packed, _ = self.rnn(pack_padded_sequence(h, seq_lengths.tolist(), enforce_sorted=False, batch_first=True))
        u, _ = pad_packed_sequence(_u_packed, batch_first=True, total_length=h.shape[1])  # -> [N, L, 2 * H']

        return u


class HeadlineParaGNN(nn.Module):
    def __init__(self, num_layers, in_features, out_features):
        super().__init__()
        self.graph_convs = nn.ModuleList([
            GCNConv(in_features, out_features),
            *[GCNConv(out_features, out_features) for _ in range(num_layers-1)]
        ])

    def forward(self, h_headline, h_para, graph, edge_weight=None):
        # merge headline and para dim: [N, 1, H], [N, P, H] -> [N, 1+P, H]
        pre_cat_dims = [1, h_para.shape[1]]
        x = torch.cat([h_headline.unsqueeze(dim=1), h_para], dim=1)

        # flatten batch and para: [N, 1+P, H] -> [N x (1+P), H]
        pre_flattened_dims = x.shape[0:2]
        x = x.flatten(0, 1)

        outs = []
        for gconv in self.graph_convs:
            prev_x = x
            x = gconv(x, graph.edge_index, edge_weight)
            x = F.relu(x + prev_x)
            outs.append(x)
        x = torch.cat(outs, dim=-1)

        # unflatten batch and para: [N x (1+P), H] -> [N, 1+P, H]
        x = x.reshape(*pre_flattened_dims, -1)

        # unmerge headline and para dim: [N, 1+P, H] -> [N, 1, H], [N, P, H]
        h_headline, h_para = torch.split(x, pre_cat_dims, dim=1)  # [N, 1, H], [N, P, H]
        return h_headline, h_para


class GlobalLocalFusionBlock(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_dim)

    def forward(self, h_headline, h_para, para_mask):
        # [N, H], [N, P, H], [N, P]
        h_para = F.relu(self.linear(h_para))
        h_headline = F.relu(self.linear(h_headline))

        h_para_masked_zero = (h_para * para_mask.to(torch.float).unsqueeze(dim=-1))
        h_global_mean = torch.cat([h_headline, h_para_masked_zero], dim=1).mean(dim=1)
        h_global_mean = h_global_mean.unsqueeze(dim=1)

        h_para_masked_neg_inf = h_para.clone()
        h_para_masked_neg_inf[~para_mask] = -float('inf')
        h_global_max, _ = torch.cat([h_headline, h_para_masked_neg_inf], dim=1).max(dim=1)
        h_global_max = h_global_max.unsqueeze(dim=1)

        v_headline = torch.cat([h_headline, h_global_mean, h_global_max], dim=-1)
        v_para = torch.cat([h_para, h_global_mean.expand_as(h_para), h_global_max.expand_as(h_para)], dim=-1)

        return v_headline, v_para


class ParaEdgeLearner(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, h_para, h_hidden):
        # In: [N, P, H], [N, H]
        # Out: [N, P]
        e = self.bilinear(h_para.contiguous(), h_hidden.unsqueeze(dim=1).expand_as(h_para).contiguous())
        # => [N, H, 1]
        return torch.sigmoid(e.squeeze(dim=2))


class GHDE(nn.Module):
    @staticmethod
    def param_args():
        return [
            ('--embed-dim', int, 300),
            ('--headline-para-dim', int, 200),
            ('--gnn-dim', int, 200),
            ('--num-gnn-layers', int, 3),
            ('--fusion-linear-dim', int, 200),
            ('--mlp-dim', int, 100),
            ('--edge-supervision', bool, True),
            ('--para-level-supervision', bool, True),
        ]

    def __init__(self, params, pretrained_embeds=None, vocab_size=None):
        super().__init__()

        # hidden dim definitions
        e_dim = params['embed_dim']
        h_dim = params['headline_para_dim']
        g_dim = params['gnn_dim']
        K = params['num_gnn_layers']
        f_dim = params['fusion_linear_dim']
        m_dim = params['mlp_dim']
        self.edge_supervision = params['edge_supervision']
        self.para_level_supervision = params['para_level_supervision']

        if pretrained_embeds is not None:
            self.word_embed = nn.Embedding.from_pretrained(pretrained_embeds)  # => h_dim
        else:
            self.word_embed = nn.Embedding(vocab_size, e_dim)  # => h_dim

        self.headline_encoder = HeadlineEncoder(e_dim, h_dim)  # => h_dim
        self.para_encoder = BodyEncoder(e_dim, h_dim)  # => h_dim
        self.inter_para_encoder = SequenceEncoder(h_dim, h_dim // 2)  # => h_dim

        self.para_edge_nn = ParaEdgeLearner(h_dim)
        self.headline_para_gnn = HeadlineParaGNN(K, h_dim, g_dim)  # => K * g_dim

        self.fusion_block = GlobalLocalFusionBlock(K * g_dim, f_dim)  # => 3 * f_dim
        self.mlp = nn.Sequential(
            nn.Linear(3 * f_dim, f_dim),
            nn.ReLU(),
            nn.Linear(f_dim, m_dim),
            nn.ReLU(),
        )  # => m_dim
        self.bilinear = nn.Bilinear(m_dim, m_dim, 1)

    def forward(self, inputs):
        x_headline = self.word_embed(inputs['headline'])  # [N, L_headline, H]
        x_para = self.word_embed(inputs['paras'])  # [N, P, L_para, H]
        para_mask = inputs['para_lengths'] != 0

        h_headline = self.headline_encoder(x_headline, inputs['headline_lengths'])
        h_para = self.para_encoder(x_para, inputs['para_lengths'], inputs['num_para_in_body'])
        v_para = self.inter_para_encoder(h_para, inputs['num_para_in_body'])

        if self.edge_supervision:
            edge_weight_matrix = self.para_edge_nn(v_para, h_headline)
            symmetric_edge_weight = torch.cat([edge_weight_matrix, edge_weight_matrix], dim=1)
            symmetric_edge_weight = symmetric_edge_weight[torch.cat([para_mask, para_mask], dim=1)]
        else:
            symmetric_edge_weight = None

        u_headline, u_para = self.headline_para_gnn(h_headline, v_para, inputs['graph'], symmetric_edge_weight)
        # [N, 1, H], [N, P, H]

        h_headline, h_para = self.fusion_block(u_headline, u_para, para_mask)
        h_headline = self.mlp(h_headline)
        h_para = self.mlp(h_para)
        para_logits = self.bilinear(h_headline.expand_as(h_para).contiguous(), h_para.contiguous()).squeeze(dim=-1)
        para_logits_masked_neginf = para_logits.clone()
        para_logits_masked_neginf[inputs['para_lengths'] == 0] = -float('inf')
        doc_logits, _ = para_logits_masked_neginf.max(dim=1)

        outputs = {
            'doc_logits': doc_logits
        }
        if self.para_level_supervision:
            outputs['para_logits'] = para_logits
        if self.edge_supervision:
            outputs['para_edge_weights'] = edge_weight_matrix

        return outputs
