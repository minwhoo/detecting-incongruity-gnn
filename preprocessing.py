# -*- coding: utf-8 -*-

from collections import Counter
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv

MAX_HEADLINE_LEN = 21
MAX_TEXT_LEN = 2200
MAX_NUM_SENT = 40
MAX_NUM_PARA = 170
VOCA_MIN_FREQ = 8


def sent_to_seq(sent, voca):
    unk_idx = voca['<unk>']
    return [voca.get(token, unk_idx) for token in sent.lower().strip().split()]


def parse_headline(headline_list, voca):
    return [sent_to_seq(h, voca) for h in tqdm(headline_list, desc="Convert headline tokens into indices")]


def parse_body(body_list, voca):
    bodys = []
    eop_idx = voca['<eop>']
    eos_idx = voca['<eos>']
    empty_body_indices = []
    for i, b in enumerate(tqdm(body_list, desc="Parse body into structured document of indices")):
        body = []
        sent = []
        para = []
        for idx in sent_to_seq(b, voca):
            if idx != eos_idx and idx != eop_idx:
                sent.append(idx)
            elif idx == eos_idx:
                if len(sent) != 0:
                    para.append(sent)
                    sent = []
            elif idx == eop_idx:
                if len(para) != 0:
                    body.append(para)
                    para = []
        if len(sent) != 0:
            para.append(sent)
        if len(para) != 0:
            body.append(para)

        if len(body) != 0:
            bodys.append(body)
        else:
            empty_body_indices.append(i)
    return bodys, empty_body_indices


def merge_body_into_seq(bodys, voca):
    texts = []
    sent_start_idxs = []
    sent_lens = []
    sent_para_mappings = []
    para_start_idxs = []
    para_lens = []
    eop_idx = voca['<eop>']
    eos_idx = voca['<eos>']

    for body in tqdm(bodys, desc="Flatten structured document into sequence"):
        text = []
        sent_start_idx = []
        sent_len = []
        sent_para_mapping = []
        para_start_idx = []
        para_len = []
        for i, para in enumerate(body):
            para_start_idx.append(len(text))
            p_len = 0
            for sent in para:
                sent_start_idx.append(len(text))
                sent_len.append(len(sent))
                sent_para_mapping.append(i)
                text.extend(sent)
                text.append(eos_idx)
                p_len += len(sent) + 1
            para_len.append(p_len)
            if i != len(body) - 1:
                text.append(eop_idx)
        texts.append(text)
        sent_start_idxs.append(sent_start_idx)
        sent_lens.append(sent_len)
        sent_para_mappings.append(sent_para_mapping)
        para_start_idxs.append(para_start_idx)
        para_lens.append(para_len)
    return texts, sent_start_idxs, sent_lens, sent_para_mappings, para_start_idxs, para_lens


def seq_to_numpy(seq, max_seq_len, pad_val=0, dtype=np.int32):
    fixed_len_seq = []
    for _, s in enumerate(tqdm(seq, desc="Pad sequence and stack as np array")):
        pad_count = max_seq_len - len(s)
        if pad_count > 0:
            s = s + [pad_val] * pad_count
        fixed_len_seq.append(s[:max_seq_len])
    return np.asarray(fixed_len_seq, dtype=dtype)


def load_data_as_df(dataset_path):
    column_names = ['id', 'headline', 'body', 'label', 'fake_para_len', 'fake_para_index', 'fake_type']
    df = pd.read_csv(dataset_path, sep='\t', header=None, names=column_names)
    return df


def parse_para_label(fake_para_index, max_num_para):
    # convert NaN values to empty list string
    para_index = fake_para_index.copy()
    para_index[para_index.isna()] = "[]"

    para_labels = np.zeros((len(para_index), max_num_para), dtype=np.bool)
    for i, list_string in enumerate(para_index):
        # parse string representation of list to python list
        indices = [int(x.strip())for x in list_string[1:-1].split(',')  if x != '']
        # remove indices over max para
        indices = [x for x in indices if x < max_num_para]
        # set para
        para_labels[i, indices] = True
    return para_labels


def compute_voca(df, min_freq_threshold):
    token_counter = Counter()

    for h in tqdm(df.headline, desc="Count all tokens in headline"):
        for token in h.lower().strip().split():
            token_counter[token] += 1

    for b in tqdm(df.body, desc="Count all tokens in body"):
        for token in b.lower().strip().split():
            token_counter[token] += 1

    voca = {
        '': 0,
        '<unk>': 1
    }

    idx = 2
    for token, freq in reversed(token_counter.most_common()):
        if freq >= min_freq_threshold:
            voca[token] = idx
            idx += 1

    return voca


def load_glove_dict(path):
    glove_dict = {}
    with open(path, 'r') as f:
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            embedding = [float(v) for v in tokens[1:]]
            glove_dict[word] = embedding
    return glove_dict


def get_voca_embedding(voca, embed_dict):
    embed_size = len(next(iter(embed_dict.values())))
    voca_embedding = []
    missing_cnt = 0
    for token in voca:
        if token == '':
            voca_embedding.append(np.zeros(embed_size))
        elif token in embed_dict:
            voca_embedding.append(embed_dict[token])
        else:
            voca_embedding.append(np.random.uniform(-0.25, 0.25, embed_size).tolist())
            missing_cnt += 1

    print("Embedding coverage: {:.5f}".format(1.0 - (missing_cnt/float(len(voca)))))
    return np.asarray(voca_embedding, dtype=np.float32)


def process_dataset(df, voca):
    print("STEP 1/5: Convert body data into structured document of indices")
    bodys, empty_body_indices = parse_body(df.body, voca)

    if len(empty_body_indices) != 0:
        print("{} samples with empty body found!".format(len(empty_body_indices)))
        print("STEP 2/5: Remove samples with empty body")
        df.drop(empty_body_indices, inplace=True)

    print("STEP 3/5: Convert headline tokens to indices")
    headline_seq = parse_headline(df.headline, voca)

    print("STEP 4/5: Flatten back document while computing metadata")
    body_seq, sent_start_idx, sent_len, sent_para_mapping, para_start_idx, para_len = merge_body_into_seq(bodys, voca)

    print("STEP 5/5: Pad and convert all data to numpy arrays")
    np_data = {
        'id': df.id.to_numpy(),
        'headline': seq_to_numpy(headline_seq, MAX_HEADLINE_LEN),
        'body': seq_to_numpy(body_seq, MAX_TEXT_LEN),
        'label': df.label.to_numpy(),
        'sent_start_idx': seq_to_numpy(sent_start_idx, MAX_NUM_SENT, dtype=np.int16),
        'sent_len': seq_to_numpy(sent_len, MAX_NUM_SENT, dtype=np.int16),
        'sent_para_mapping': seq_to_numpy(sent_para_mapping, MAX_NUM_SENT, pad_val=-1, dtype=np.int16),
        'para_start_idx': seq_to_numpy(para_start_idx, MAX_NUM_PARA, dtype=np.int16),
        'para_len': seq_to_numpy(para_len, MAX_NUM_PARA, dtype=np.int16)
    }

    if 'fake_para_index' in df.columns:
        np_data['para_label'] = parse_para_label(df.fake_para_index, MAX_NUM_PARA)

    num_samples = len(body_seq)
    # check first dimension of all data is num_samples
    for d in np_data.values():
        assert len(d) == num_samples

    return np_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True, help="Dataset path")
    parser.add_argument("--glove-path", type=Path,
        help="Path for GloVe embeddings to load and extract embeddings for the voca of (train) dataset")
    parser.add_argument("--debug-dataset-size", type=int, default=200,
        help="Size of the debug dataset created by taking slice of dev dataset")
    args = parser.parse_args()

    if not args.dataset_path.exists():
        raise ValueError("Specified dataset path does not exist!")

    if args.glove_path is not None and not args.glove_path.exists():
        raise ValueError("Specified glove file does not exist!")

    processed_data_path = args.dataset_path / 'processed'
    if not processed_data_path.exists():
        processed_data_path.mkdir(parents=True)

    voca_save_path = processed_data_path / f"voca_mincut{VOCA_MIN_FREQ}.txt"
    voca_glove_embedding_save_path = processed_data_path / 'voca_glove_embedding.npy'

    if not processed_data_path.exists():
        processed_data_path.mkdir(parents=True)

    for dataset_type in ['train', 'dev', 'test']:
        print("======= {} =======".format(dataset_type.upper()))
        if dataset_type == 'real_world':
            save_path = processed_data_path
        else:
            save_path = processed_data_path / dataset_type
        if not save_path.exists():
            save_path.mkdir()

        #-------------      Load data      --------------
        print("Loading data...")
        df = load_data_as_df(args.dataset_path / (dataset_type + '.tsv'))

        # Compute/load voca
        if dataset_type == 'train':
            print("Computing voca...")
            voca = compute_voca(df, VOCA_MIN_FREQ)
            with open(voca_save_path, 'w') as f:
                sorted_voca = sorted(voca.items(), key=lambda x: x[1])
                for voc, idx in sorted_voca:
                    f.write(f"{voc}\n")
            print("Saved voca with {} tokens!".format(len(voca)))

        #-------------    Process data     --------------
        print("Processing data...")
        np_data = process_dataset(df, voca)

        # Save processed data to file
        print("Saving data...")
        for name, data in np_data.items():
            np.save(save_path / '{}.npy'.format(name), data)

        #------------- Auxiliary operations ----------------

        if dataset_type == 'train':
            # Save glove embeddings
            if args.glove_path is not None:
                print("Fetching glove embeddings...")
                glove_dict = load_glove_dict(args.glove_path)
                embedding = get_voca_embedding(voca, glove_dict)
                np.save(voca_glove_embedding_save_path, embedding)
                del glove_dict
                del embedding
        elif dataset_type == 'dev':
            # Save debug dataset by taking slice of dev set
            print("Saving debug data...")
            debug_save_path = processed_data_path / 'debug'
            if not debug_save_path.exists():
                debug_save_path.mkdir()
            for name, data in np_data.items():
                np.save(debug_save_path / '{}.npy'.format(name), data[:args.debug_dataset_size])

        # free up memory
        del df
        del np_data


if __name__ == "__main__":
    main()
