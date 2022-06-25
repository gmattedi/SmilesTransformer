import logging
from typing import Dict, List

import torch

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('ROOT')


def output2smiles(
        output: torch.Tensor,
        idx2token: Dict[int, str],
        eos_idx: int,
        drop_first: int = 1,

) -> List[str]:
    batch_smi = []
    for entry in output:
        entry_tokens = []

        for val in entry[drop_first:]:
            if val == eos_idx:
                break
            else:
                entry_tokens.append(idx2token[val])
        batch_smi.append(''.join(entry_tokens))
    return batch_smi
