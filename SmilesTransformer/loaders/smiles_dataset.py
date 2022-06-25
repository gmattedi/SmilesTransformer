from numbers import Number
from typing import Optional

import pandas as pd
import torch.utils.data

from SmilesTransformer.loaders.base import SeqDataset, paired_collate_fn, augment_smiles
from SmilesTransformer.tokenizer import load_mapping, RegexTokenizer, dense_onehot


def build_loader(
        csv_path: str,
        src_col: str, tgt_col: str,
        alphabet_path: str,
        sample: Optional[Number] = None,
        random_state: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 1,
        augment_times: int = 0,
        csv_sep: str = ',',
        alphabet_sep: str = '\n'
):
    """
    Build torch dataloader from SMILES file
    Args:
        csv_path (str): CSV/CSV.GZ-like file with SMILES strings
        src_col (str): Column with source SMILES
        tgt_col (str): Column with target SMILES
        alphabet_path (str): File containing the alphabet.
            Generate it with SmilesTransformer.tokenizer.tokenizer
        sample (Optional[Number]): Take a sample of size `sample` of the CSV file
        random_state (Optional[int]): Random state for sampling the CSV file
        batch_size (int)
        num_workers (int)
        augment_times (int): Augment SMILES `augment_times` times by generating non-canonical SMILES.
            Note that this *WILL shuffle* the order of the SMILES pairs (inter-pair ordering, not intra-pair).
            Note that the final number of pairs is likely to be less than 1 + augment_times as the same non-canonical
            smiles may be generated more than once
        csv_sep (str): CSV file separator
        alphabet_sep (str): Alphabet file separator
    Returns:
        loader (data.DataLoader), token2idx (Dict[str, int]), idx2token (Dict[int, str])

    """
    tokenizer = RegexTokenizer()

    token2idx, idx2token = load_mapping(alphabet_path, sep=alphabet_sep)

    df = pd.read_csv(csv_path, sep=csv_sep)
    if sample is not None:
        df = df.sample(sample, random_state=random_state)

    src_smiles, tgt_smiles = df[src_col].values, df[tgt_col].values

    if augment_times > 0:
        pairs = set(zip(src_smiles, tgt_smiles))
        for _ in range(augment_times):
            pairs_aug = {
                (augment_smiles(pair[0]), augment_smiles(pair[1]))
                for pair in pairs
            }
            pairs.update(pairs_aug)

        src_smiles, tgt_smiles = list(zip(*list(pairs)))

    src_tokens = [tokenizer.tokenize(smi) for smi in src_smiles]
    tgt_tokens = [tokenizer.tokenize(smi) for smi in tgt_smiles]

    src_onehot = [dense_onehot(t, token2idx) for t in src_tokens]
    tgt_onehot = [dense_onehot(t, token2idx) for t in tgt_tokens]

    loader = torch.utils.data.DataLoader(
        SeqDataset(
            src_word2idx=token2idx,
            tgt_word2idx=token2idx,
            src_insts=src_onehot,
            tgt_insts=tgt_onehot),
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    return loader, token2idx, idx2token
