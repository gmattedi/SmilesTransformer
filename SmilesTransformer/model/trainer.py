"""
This script handling the training process.
"""
import logging
import math
import os
import time
from typing import Union, List

import torch
from torch.utils import data
from tqdm import tqdm

import SmilesTransformer.model.transformer.Constants as Constants
from SmilesTransformer.model.eval import eval_performance
from SmilesTransformer.utils import logger


def train(
        model: torch.Module, train_loader: data.DataLoader,
        val_loader: data.DataLoader, optimizer,
        device: Union[torch.device, str],
        label_smoothing: bool = False,
        n_epochs: int = 10,
        checkpoint_folder: str = "./",
        logger: logging.Logger = logger
) -> List[List[float]]:
    """
    Train the transformer

    Args:
        model (torch.Module): Transformer
        train_loader (DataLoader):
        val_loader (DataLoader):
        optimizer:
        device (Union[torch.device, str]):
        label_smoothing (bool):
        n_epochs (int):
        checkpoint_folder (str):
        logger (logging.logger): Logger

    Returns:
        history (List[List[float]]): Train/val loss and accuracy.
            One list per epoch, in the form [train_loss, train_acc, valid_loss, valid_acc]

    """
    history = []
    valid_accus = []
    for epoch_i in range(n_epochs):
        logger.info(f"[ Epoch {epoch_i} ]")

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, train_loader, optimizer, device, smoothing=label_smoothing)
        logger.info("(Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, time: {time:3.3f} min".format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
            time=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, val_loader, device)
        logger.info("(Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, time: {time:3.3f} min".format(
            ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
            time=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "epoch": epoch_i}

        model_path = os.path.join(checkpoint_folder, "model.ckpt")
        if valid_accu >= max(valid_accus):
            torch.save(checkpoint, model_path)
            logger.info("Checkpoint file updated")

        history.append([
            train_loss, train_accu, valid_loss, valid_accu
        ])

    return history


def train_epoch(model, training_data, optimizer, device, smoothing):
    """ Epoch operation in training phase"""

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc="(Training)   ", leave=False):
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = eval_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    """ Epoch operation in evaluation phase """

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc="(Validation) ", leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = eval_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy
