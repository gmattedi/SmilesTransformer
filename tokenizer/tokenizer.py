import re
from typing import Sequence, Dict, List, Tuple

import numpy as np

from transformer import Constants

# TODO cite
SMI_REGEX_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


class RegexTokenizer:
    """Run regex tokenization"""

    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN) -> None:
        """Constructs a RegexTokenizer.
        Args:
            regex_pattern: regex pattern used for tokenization.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, text: str, bos_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD) -> List[str]:
        """Regex tokenization.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens separated by spaces.
        """
        tokens = [token for token in self.regex.findall(text)]
        tokens = [bos_token] + tokens + [eos_token]
        return tokens


def build_vocabulary(smiles_lst: Sequence[str]) -> Dict[str, int]:
    tokenizer = RegexTokenizer()

    tokens = [tokenizer.tokenize(smi) for smi in smiles_lst]
    unique_tokens, unique_counts = np.unique(np.concatenate(tokens), return_counts=True)

    token2idx = {}

    i = 0
    for t, c in zip(unique_tokens, unique_counts):
        if c <= 5:
            continue
        else:
            token2idx[t] = i
            i += 1

    special_tokens = [Constants.BOS_WORD, Constants.EOS_WORD, Constants.PAD_WORD, Constants.UNK_WORD]
    for t in special_tokens:
        if t not in token2idx:
            token2idx[t] = token2idx.get(t, i)
            i += 1

    return token2idx


def load_mapping(alphabet_path: str, sep: str = '\n') -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(alphabet_path) as handle:
        alphabet = handle.read().split(sep)

    token2idx = {token: i for i, token in enumerate(alphabet)}
    idx2token = {i: token for token, i in token2idx.items()}

    return token2idx, idx2token


def dense_onehot(tokens, token2idx: Dict[str, int]):
    """Mapping words to idx sequence"""
    return [token2idx.get(w, token2idx[Constants.UNK_WORD]) for w in tokens]


if __name__ == '__main__':
    import pandas as pd

    chembl = pd.read_csv('../data/chembl_30/chembl_30_chemreps_proc_train.csv.gz').sample(100)
    smiles = chembl.SMILES.values

    token2idx = build_vocabulary(smiles)
    idx2token = {i: token for token, i in token2idx.items()}
    vocab = [idx2token[i] for i in range(len(idx2token))]

    with open('alphabet.dat', 'wt') as handle:
        handle.write('\n'.join(vocab))
