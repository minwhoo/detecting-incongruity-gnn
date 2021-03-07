# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

PAD_TOKEN = 0


def split_batch_text_into_segments(text, seg_start_idx, seg_end_idx, max_seg_len):
    """Split batch of text into to segments given start & end idx of each segment."""
    # create col index matrix that maps from [*, T] to [*, S, L]
    token_indices = torch.arange(max_seg_len).reshape([1, 1, -1])  # [1, 1, L]
    col = token_indices + seg_start_idx.unsqueeze(-1)  # [N, S, L]
    col[col >= seg_end_idx.unsqueeze(-1)] = -1  # set index of indices past last index as last

    # create index matrix that maps each [N, *] to [N, *]
    row = torch.arange(len(text)).reshape(-1, 1, 1)
    # -> [N, 1, 1] (automatically broadcasted to [B, S, L] during indexing)

    # add pad token at the end of every sequence to ensure all sequences end with PAD TOKEN, and -1 index will pad it
    text_padded = torch.cat([text, PAD_TOKEN * torch.ones(len(text), dtype=text.dtype).unsqueeze(-1)], dim=1)

    # create output matrix using advanced indexing
    return text_padded[row, col]


class IncongruentHeadlineNewsDataset(Dataset):
    def __init__(self, data_path: Path, seq_type, max_seq_len, construct_graph, indices=None):
        super().__init__()
        assert seq_type in ["sent", "para"]
        self.seq_type = seq_type
        self.max_seq_len = max_seq_len
        self.construct_graph = construct_graph

        # load data
        p = data_path

        self.headline = torch.tensor(np.load(p / "headline.npy"), dtype=torch.long)
        self.headline_lengths = (self.headline != PAD_TOKEN).sum(dim=1)
        self.body = torch.tensor(np.load(p / "body.npy"), dtype=torch.long)
        self.label = torch.tensor(np.load(p / "label.npy"), dtype=torch.float)
        if (p / "para_label.npy").exists():
            self.para_label = torch.tensor(np.load(p / "para_label.npy"), dtype=torch.float)
        else:
            # dummy data for real-world data; shouldn't be used for training
            self.para_label = torch.empty([len(self.body), self.max_seq_len], dtype=torch.float)

        if self.seq_type == "sent":
            self.seq_start_idx = torch.tensor(np.load(p / "sent_start_idx.npy"), dtype=torch.long)
            self.seq_len = torch.tensor(np.load(p / "sent_len.npy"), dtype=torch.long)
            self.sent_para_mapping = torch.tensor(np.load(p / "sent_para_mapping.npy"), dtype=torch.long)
        else:
            self.seq_start_idx = torch.tensor(np.load(p / "para_start_idx.npy"), dtype=torch.long)
            self.seq_len = torch.tensor(np.load(p / "para_len.npy"), dtype=torch.long)

        if indices is not None:
            self.headline = self.headline[indices]
            self.headline_lengths = self.headline_lengths[indices]
            self.body = self.body[indices]
            self.label = self.label[indices]
            self.para_label = self.para_label[indices]
            self.seq_start_idx = self.seq_start_idx[indices]
            self.seq_len = self.seq_len[indices]
            if self.seq_type == "sent":
                self.sent_para_mapping = self.sent_para_mapping[indices]

    def __len__(self):
        return len(self.headline)

    def __getitem__(self, idx):
        data = {
            'headline': self.headline[idx],
            'headline_lengths': self.headline_lengths[idx],
            'body': self.body[idx],
            'label': self.label[idx],
            'para_label': self.para_label[idx],
            'seq_start_idx': self.seq_start_idx[idx],
            'seq_len': self.seq_len[idx]
        }
        if self.seq_type == "sent":
            data['sent_para_mapping'] = self.sent_para_mapping[idx]
        return data

    def collate(self, data_list):
        batch_data = {k: torch.stack([d[k] for d in data_list], dim=0) for k in data_list[0].keys()}
        out_data = {
            'headline': batch_data['headline'],
            'headline_lengths': batch_data['headline_lengths'],
            'label': batch_data['label'],
            'para_label': batch_data['para_label']
        }

        seq_len = batch_data['seq_len']
        seq_start_idx = batch_data['seq_start_idx']

        # truncate max seq len
        max_seq_len = min(self.max_seq_len, seq_len.max().item())

        # compute seq end idx
        seq_len[seq_len > max_seq_len] = max_seq_len
        seq_end_idx = seq_start_idx + seq_len

        # ignore sequence that are out-of-bounds of original text
        seq_out_of_bounds = seq_end_idx >= batch_data['body'].shape[1]
        seq_start_idx[seq_out_of_bounds] = 0
        seq_end_idx[seq_out_of_bounds] = 0
        seq_len[seq_out_of_bounds] = 0

        # truncate max num seq
        max_num_seq = (seq_len != 0).sum(dim=1).max().item()
        seq_start_idx = seq_start_idx[:, :max_num_seq]
        seq_len = seq_len[:, :max_num_seq]
        seq_end_idx = seq_end_idx[:, :max_num_seq]

        seqs = split_batch_text_into_segments(batch_data['body'], seq_start_idx, seq_end_idx, max_seq_len)

        # compute metadata
        num_seq_in_body = (seq_len != 0).sum(dim=1)
        max_num_seq_in_body = num_seq_in_body.max().item()

        if self.seq_type == "sent":
            sent_para_mapping = batch_data['sent_para_mapping']
            sent_para_mapping[seq_out_of_bounds] = -1
            sent_para_mapping = sent_para_mapping[:, :max_num_seq]

            sent_max_para_idx, _ = sent_para_mapping.max(dim=1)
            num_para_in_body = sent_max_para_idx + 1
            max_num_para_in_body = num_para_in_body.max().item()

            para_indices = torch.arange(max_num_para_in_body)
            num_sent_in_para = (para_indices.reshape([1, 1, -1]) == sent_para_mapping.unsqueeze(-1)).sum(dim=1)

            out_data['sents'] = seqs
            out_data['sent_lengths'] = seq_len
            out_data['num_sent_in_para'] = num_sent_in_para
            out_data['num_sent_in_body'] = num_seq_in_body
            out_data['num_para_in_body'] = num_para_in_body
            out_data['para_lengths'] = num_sent_in_para
            out_data['para_label'] = out_data['para_label'][:, :max_num_para_in_body]
            if self.construct_graph:
                graph_data_list = []
                for p, s, sp in zip(num_para_in_body, num_seq_in_body, num_sent_in_para):
                    graph_data = self._create_sent_para_graph(p, max_num_para_in_body, s, max_num_seq_in_body, sp)
                    graph_data_list.append(graph_data)
                batch_graph = Batch.from_data_list(graph_data_list)
                out_data['graph'] = batch_graph
        else:
            out_data['paras'] = seqs
            out_data['para_lengths'] = seq_len
            out_data['num_para_in_body'] = num_seq_in_body
            out_data['para_label'] = out_data['para_label'][:, :max_num_seq_in_body]
            if self.construct_graph:
                graph_data_list = [self._create_para_graph(p, max_num_seq_in_body) for p in num_seq_in_body]
                batch_graph = Batch.from_data_list(graph_data_list)
                out_data['graph'] = batch_graph

        return out_data

    def _create_para_graph(self, num_para, max_num_para):
        para_nodes = torch.arange(num_para) + 1
        headline_nodes = torch.zeros_like(para_nodes)
        edge_index = torch.stack([
            headline_nodes,
            para_nodes
        ])  # -> [2, P]

        # para_edge_variant = "fully-connected"
        para_edge_variant = None  # IMPORTANT: To use edge variant other than default, must set edge supervision arg to False
        if para_edge_variant == "neighbor":
            para_left_nodes = para_nodes[:-1]
            para_right_nodes = para_nodes[1:]
            para_edge_index = torch.stack([
                para_left_nodes,
                para_right_nodes
            ])  # -> [2, P]
            edge_index = torch.cat([edge_index, para_edge_index], dim=1)  # -> [2, 2P]
        elif para_edge_variant == "fully-connected":
            para_edges = []
            for i in range(1, num_para + 1):
                for j in range(i+1, num_para + 1):
                    para_edges.append([i,j])
            para_edge_index = torch.tensor(para_edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, para_edge_index], dim=1)  # -> [2, 2P]

        # add reverse edges to make edges undirected
        reversed_edge_index = edge_index.roll(1, dims=0)
        edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)

        data = Data(edge_index=edge_index)
        data.num_nodes = max_num_para + 1
        return data

    def _create_sent_para_graph(self, num_para, max_num_para, num_sents_in_body, max_num_sents_in_body,
                                num_sents_in_para):
        # P', P, S', S, [P]
        # sent-para edges
        sent_idx = num_sents_in_para[num_sents_in_para != 0].cumsum(dim=0)
        sent_para_idx = torch.zeros(num_sents_in_body, dtype=torch.long)
        sent_para_idx[sent_idx[:-1]] = 1
        sent_para_idx = sent_para_idx.cumsum(dim=0) + 1
        sent_idx = torch.arange(num_sents_in_body) + 1 + max_num_para
        sent_para_edges = torch.stack([sent_para_idx, sent_idx])

        # sent-para edges
        para_idx = torch.arange(num_para) + 1
        para_headline_idx = torch.zeros(num_para, dtype=torch.long)
        para_headline_edges = torch.stack([para_idx, para_headline_idx])

        edge_index = torch.cat([para_headline_edges, sent_para_edges], dim=-1)
        reversed_edge_index = edge_index.roll(1, dims=0)

        data = Data(edge_index=torch.cat([edge_index, reversed_edge_index], dim=-1))
        data.num_nodes = max_num_sents_in_body + max_num_para + 1
        return data
