import math

import numpy as np
import torch.utils.data

import trainer
import transformer.Constants as Constants
from trainer import train
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from loaders import build_loader

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
import tqdm as t

RDLogger.DisableLog('rdApp.*')

from utils import logger

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Device: {device}')


logger.info('Creating training loader')
train_loader, token2idx, idx2token = build_loader(
    csv_path='data/chembl_30/chembl_30_chemreps_proc_train.csv.gz',
    src_col='SMILES', tgt_col='SMILES', alphabet_path='tokenizer/alphabet.dat',
    sample=100000, random_state=123, batch_size=64, num_workers=10, augment_times=1
)

logger.info('Creating validation loader')
val_loader, _, _ = build_loader(
    csv_path='data/chembl_30/chembl_30_chemreps_proc_val.csv.gz',
    src_col='SMILES', tgt_col='SMILES', alphabet_path='tokenizer/alphabet.dat',
    sample=5000, random_state=123, batch_size=64, num_workers=10, augment_times=1
)

logger.info('Instantiating the transformer')
transformer = Transformer(
    n_src_vocab=len(token2idx),
    n_tgt_vocab=len(token2idx),
    len_max_seq=100,
    d_word_vec=512, d_model=512, d_inner=2048,
    n_layers=6, n_head=8,
    d_k=64, d_v=64,
    dropout=0.1,
    tgt_emb_prj_weight_sharing=True,
    emb_src_tgt_weight_sharing=True
).to(device)
logger.info(transformer)

optimizer = ScheduledOptim(
    torch.optim.Adam(
        filter(lambda x: x.requires_grad, transformer.parameters()),
        betas=(0.9, 0.98), eps=1e-09),
    512, 4000)

### Evaluation with untrained model

val_loss, val_acc = trainer.eval_epoch(transformer, val_loader, device)
print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %'.format(
    ppl=math.exp(min(val_loss, 100)), accu=100 * val_acc))

# Train

history = train(transformer, train_loader, val_loader, optimizer, device, n_epochs=10)

# Evaluate

_ = transformer.eval()


def output2smiles(output: torch.Tensor, token2idx=token2idx, drop_first: int = 1):
    eos_idx = token2idx[Constants.EOS_WORD]

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


output_smi = []

for batch in t.tqdm(val_loader):
    src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
    gold = tgt_seq[:, 1:]

    with torch.no_grad():
        pred = transformer(src_seq, src_pos, tgt_seq, tgt_pos).cpu().detach()
        pred = pred.max(1)[1].reshape(gold.shape).numpy()

    output_smi = output_smi + output2smiles(pred, drop_first=0)

output_mol = list(map(Chem.MolFromSmiles, output_smi))
output_mol_valid = list(filter(None, output_mol))

fraction_valid = len(output_mol_valid) / len(output_mol)

msg = f'{len(output_mol_valid)}/{len(output_mol)} ({fraction_valid * 100:.2f} %)'
print(msg)

Draw.MolsToGridImage(
    np.random.choice(output_mol_valid, 24, replace=False),
    molsPerRow=8
)
