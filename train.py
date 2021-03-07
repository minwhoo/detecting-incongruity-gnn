# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import time
import datetime

import torch
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models import GHDE
from dataset import IncongruentHeadlineNewsDataset
import utils
import preprocessing


def move_batch_dict_to_device(batch, device):
    inputs = {}
    for k, v in batch.items():
        if k == 'graph':
            v.batch = v.batch.to(device)
            v.edge_index = v.edge_index.to(device)
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
        else:
            inputs[k] = v
    return inputs


def compute_loss(inputs, outputs, criterion, edge_criterion):
    """Compute loss automatically based on what the model output dict contains.
    'doc_logits' -> document-label CE loss
    'para_logits' -> paragraph-label CE loss
    'pare_edge_weights' -> additional edge CE loss
    """
    if 'para_logits' in outputs:
        loss = criterion(outputs['para_logits'], inputs['para_label'])
        loss *= (inputs['para_lengths'] != 0).to(torch.float)
        loss = loss.sum()
    else:
        loss = criterion(outputs['doc_logits'], inputs['label']).sum()

    if 'para_edge_weights' in outputs:
        edge_loss_lambda = 0.1
        edge_loss = edge_criterion(outputs['para_edge_weights'], 1 - inputs['para_label'])
        edge_loss *= (inputs['para_lengths'] != 0).to(torch.float)
        loss += edge_loss_lambda * edge_loss.sum()

    return loss


def evaluate(model, dataset, device):
    model.eval()

    with torch.no_grad():
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        edge_criterion = nn.BCELoss(reduction='none')

        eval_batch_size = 64
        val_dataloader = DataLoader(dataset, batch_size=eval_batch_size, collate_fn=dataset.collate)

        loss_sum = 0
        para_num_correct = 0
        para_num_total = 0
        preds_all = []
        labels_all = []
        for batch in val_dataloader:
            inputs = move_batch_dict_to_device(batch, device)

            outputs = model(inputs)

            loss = compute_loss(inputs, outputs, criterion, edge_criterion)
            loss_sum += loss

            # doc-level preds
            preds_all.append(outputs['doc_logits'])
            labels_all.append(inputs['label'])

            # para-level preds
            if 'para_logits' in outputs:
                para_preds = outputs['para_logits']
                para_mask = (inputs['para_lengths'] != 0).to(torch.float)
                batch_num_correct = torch.eq((para_preds > 0).int(), inputs['para_label'].int()).to(torch.float)
                para_num_correct += (batch_num_correct * para_mask).sum().item()
                para_num_total += para_mask.sum().item()


        preds_all = torch.cat(preds_all)
        labels_all = torch.cat(labels_all)

        num_correct = torch.eq((preds_all > 0).int(), labels_all.int()).sum().item()
        num_total = len(preds_all)

        if para_num_total == 0:
            loss = loss_sum / num_total
            para_acc = 0
        else:
            loss = loss_sum / para_num_total
            para_acc = para_num_correct / para_num_total
        acc = num_correct / num_total
        auc = roc_auc_score(labels_all.tolist(), preds_all.tolist())

        return loss, para_acc, acc, auc


def predict(model, dataset, device, return_paragraph_predictions=False):
    model.eval()

    with torch.no_grad():
        pred_batch_size = 64
        dataloader = DataLoader(dataset, batch_size=pred_batch_size, collate_fn=dataset.collate)

        preds_all = []
        preds_para = []
        for batch in dataloader:
            inputs = move_batch_dict_to_device(batch, device)

            outputs = model(inputs)
            preds_all.append(torch.sigmoid(outputs['doc_logits']))

            # return paragraph-level sigmoid outputs
            # only works for models that output 'para_logits'
            if return_paragraph_predictions:
                para_preds = torch.sigmoid(outputs['para_logits'])
                para_preds[inputs['para_lengths'] == 0] = -1  # mask logits for padding tokens
                for para_pred in para_preds:
                    preds_para.append([p.item() for p in para_pred if p.item() != -1])

        preds_all = torch.cat(preds_all).tolist()

    return preds_all, preds_para


def train(model, train_dataset, val_dataset, args, device):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    edge_criterion = nn.BCELoss(reduction='none')

    num_iterations_per_epoch = len(train_dataset) / args.batch_size
    val_eval_freq = max(int(args.val_evaluation_freq * num_iterations_per_epoch), 1)
    print(f"Val set evaluated every {val_eval_freq:,} steps (approx. {args.val_evaluation_freq} epoch)")

    es = utils.EarlyStopping(args.early_stopping_patience)
    initial_time = time.time()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train_dataset.collate)

    global_step = 0
    epoch_no = 0
    while True:
        print(f"EPOCH #{epoch_no+1}")

        # Train single epoch
        for batch in train_dataloader:
            optimizer.zero_grad()

            inputs = move_batch_dict_to_device(batch, device)

            outputs = model(inputs)

            loss = compute_loss(inputs, outputs, criterion, edge_criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            global_step += 1

            if global_step % val_eval_freq == 0:
                # Evaluate on validation set
                val_loss, val_para_acc, val_acc, val_auc = evaluate(model, val_dataset, device)
                model.train()

                end_time = time.time()
                minutes_elapsed = int((end_time - initial_time)/60)
                print("STEP: {:7} | TIME: {:4}min | TRAIN LOSS: {:.4f} | VAL LOSS: {:.4f} | VAL ACC(PARA): {:.4f} | VAL ACC(DOC): {:.4f} | VAL AUROC(DOC): {:.4f}".format(
                    global_step, minutes_elapsed, loss, val_loss, val_para_acc, val_acc, val_auc
                ))

                # Check early stopping
                if global_step >= args.min_iterations:
                    es.record_loss(val_loss, model)

                if es.should_stop():
                    print(f"Early stopping at STEP: {global_step}")
                    print(f"Loading model checkpoint with min val loss...")
                    model.load_state_dict(torch.load(es.checkpoint_save_path))
                    es.checkpoint_save_path.unlink()  # delete temporary model checkpoint after loading
                    print(f"Loading done!")
                    return

            if global_step == args.max_iterations:
                print(f"Stopping after reaching max iteration STEP: {global_step}")
                return
        epoch_no += 1
        scheduler.step()


def main():
    # Parse Args ---------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # dataset-related params
    parser.add_argument("--processed-data-path", type=str, help="Processed dataset path")
    parser.add_argument("--debug", action="store_true")

    # model-related params
    parser.add_argument("--model", type=str, default='ghde')

    # training-related params
    parser.add_argument("--train", action="store_true",
                        help="Train model.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-iterations", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=1000000)
    parser.add_argument("--train-index-file", type=Path, help="Text file for using sampled train dataset")
    parser.add_argument("--val-evaluation-freq", type=float, default=0.1,
                        help="validation loss/acc evaluation frequency in terms of fraction of epoch")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="number of times to wait for lower val loss before early stopping")
    parser.add_argument("--no-glove", action="store_true",
                        help="train embeddings from random initialization")
    parser.add_argument("--no-cuda", action="store_true")

    # checkpoint-related params
    parser.add_argument("--load-checkpoint", type=str,
                        help="Load checkpoint from specified file. If not specified, model is randomly initialized.")
    parser.add_argument("--save-checkpoint", action='store_true',
                        help="Save checkpoint as given file name. If not specified, checkpoint is not saved")

    # eval-related params
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate model on test set.")
    parser.add_argument("--predict", action="store_true",
                        help="Predict on test sets and save predictions.")

    args, unknown = parser.parse_known_args()

    if args.debug:
        args.min_iterations = 50
        args.max_iterations = 50

    # model-specific params
    model_parser = argparse.ArgumentParser()
    for arg_name, arg_type, arg_default in GHDE.param_args():
        if arg_type is bool:
            arg_type=lambda x: (str(x).lower() in ['true','1', 'yes'])
        model_parser.add_argument(arg_name, type=arg_type, default=arg_default)
    model_args = model_parser.parse_args(unknown)

    device = torch.device('cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu')

    # Print Configuration ------------------------------------------------------
    format_string = " - {:35}: {}"
    print("{:=^70}".format(" TRAINING CONFIGURATION "))
    print("Summary")
    print("-"*70)
    print(format_string.format("device", device))
    print(format_string.format("model", args.model))
    print(format_string.format("processed_data_path", args.processed_data_path))
    print()

    print(f"Model args ({args.model})")
    print("-"*70)
    for k, v in vars(model_args).items():
        print(format_string.format(k, v))
    print()

    print("Other args")
    print("-"*70)
    for k, v in vars(args).items():
        print(format_string.format(k, v))
    print()

    print("="*70)

    # Load dataset paths and params --------------------------------------------
    print("Locating dataset...")
    processed_data_path = Path(args.processed_data_path)
    max_seq_len = preprocessing.MAX_NUM_PARA
    assert processed_data_path.exists(), f"Processed dataset not found!"
    print('Processed dataset found @ "{}" with max_seq_len={}'.format(processed_data_path, max_seq_len))

    # Initialize Model ---------------------------------------------------------
    # shared model params
    model_kwargs = {
        'params': vars(model_args)
    }

    # Load embeddings ----------------------------------------------------------
    if args.no_glove:
        voca_text_path = processed_data_path / f"voca_mincut{preprocessing.VOCA_MIN_FREQ}.txt"
        with open(voca_text_path, 'r') as f:
            vocab_size = len(f.readlines())
        model_kwargs['vocab_size'] = vocab_size
        print(f"Vocab size: {vocab_size}")
    else:
        voca_glove_embedding_path = processed_data_path / 'voca_glove_embedding.npy'
        glove_embeds = torch.tensor(np.load(voca_glove_embedding_path))
        model_kwargs['pretrained_embeds'] = glove_embeds
        print(f"Vocab size: {len(glove_embeds)}")

    print("Initializing model...")
    model = GHDE(**model_kwargs)
    model.to(device)
    print(f"model: {args.model}")

    # Load Checkpoint ----------------------------------------------------------
    if args.load_checkpoint is not None:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        model.load_state_dict(torch.load(args.load_checkpoint))

    print("Initialization done!")
    print("Model # Trainable Parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Train --------------------------------------------------------------------
    if args.train:
        print("Loading train dataset...")
        if args.train_index_file is not None:
            train_indices = []
            with open(args.train_index_file, 'r') as f:
                for line in f:
                    train_indices.append(int(line.strip()))
        else:
            train_indices = None
        train_dataset = IncongruentHeadlineNewsDataset(processed_data_path / ('train' if not args.debug else 'debug'),
                                                       seq_type='para', max_seq_len=max_seq_len, construct_graph=True,
                                                       indices=train_indices)
        print(f"Train dataset size: {len(train_dataset):9,}")

        print("Loading dev dataset...")
        dev_dataset = IncongruentHeadlineNewsDataset(processed_data_path / ('dev' if not args.debug else 'debug'),
                                                     seq_type='para', max_seq_len=max_seq_len, construct_graph=True)
        print(f"Dev dataset size: {len(dev_dataset):9,}")

        train(model, train_dataset, dev_dataset, args, device)

    # Evaluate -----------------------------------------------------------------
    if args.eval or args.predict:
        print("Loading test dataset...")
        test_dataset = IncongruentHeadlineNewsDataset(processed_data_path / ('test' if not args.debug else 'debug'),
                                                      seq_type='para', max_seq_len=max_seq_len, construct_graph=True)
        print(f"Test dataset size: {len(test_dataset):9,}")

    if args.eval:
        _, test_para_acc, test_acc, test_auc = evaluate(model, test_dataset, device)
        print(f"TEST ACC(PARA): {test_para_acc:.4f} | TEST ACC(DOC): {test_acc:.4f} | TEST AUROC(DOC): {test_auc:.4f}")

    # Predict ------------------------------------------------------------------
    if args.predict:
        preds, _ = predict(model, test_dataset, device)
        pred_record_file_name = "pred_{}.txt".format(args.model)
        with open(pred_record_file_name, "w") as f:
            for p in preds:
                f.write(f"{p}\n")
        print(f"Predictions written to file: {pred_record_file_name}")

    # Save Checkpoint ----------------------------------------------------------
    if args.save_checkpoint:
        checkpoint_file_name = f"ckpt_{args.model}_{datetime.datetime.now()}.pt".replace(' ', '_')
        print(f"Saving checkpoint to {checkpoint_file_name}")
        torch.save(model.state_dict(), checkpoint_file_name)


if __name__ == "__main__":
    main()
