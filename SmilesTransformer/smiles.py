import torch.utils.data
from rdkit import Chem, RDLogger

from SmilesTransformer.loaders import build_loader
from SmilesTransformer.model.transformer.Models import Transformer
from SmilesTransformer.model.transformer import Constants
from SmilesTransformer.model.transformer.Optim import ScheduledOptim
from SmilesTransformer.model.trainer import train
from SmilesTransformer.utils import logger
from SmilesTransformer.utils import output2smiles

RDLogger.DisableLog("rdApp.*")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device: {device}")

logger.info("Creating training loader")
train_loader, token2idx, idx2token = build_loader(
    csv_path="../data/chembl_30/chembl_30_chemreps_proc_train.csv.gz",
    src_col="SMILES", tgt_col="SMILES", alphabet_path="tokenizer/alphabet.dat",
    sample=100, random_state=123, batch_size=64, num_workers=10, augment_times=1
)

logger.info("Creating validation loader")
val_loader, _, _ = build_loader(
    csv_path="../data/chembl_30/chembl_30_chemreps_proc_valid.csv.gz",
    src_col="SMILES", tgt_col="SMILES", alphabet_path="tokenizer/alphabet.dat",
    sample=50, random_state=123, batch_size=64, num_workers=10, augment_times=1
)

logger.info("Instantiating the transformer")
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

optimizer = ScheduledOptim(
    torch.optim.Adam(
        filter(lambda x: x.requires_grad, transformer.parameters()),
        betas=(0.9, 0.98), eps=1e-09),
    512, 4000)

logger.info("Training")
history = train(transformer, train_loader, val_loader, optimizer, device, n_epochs=10)

logger.info("Training finished")

logger.info("Converting validation output to SMILES")

_ = transformer.eval()

output_smi = []

for batch in val_loader:
    src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
    gold = tgt_seq[:, 1:]

    with torch.no_grad():
        pred = transformer(src_seq, src_pos, tgt_seq, tgt_pos).cpu().detach()
        pred = pred.max(1)[1].reshape(gold.shape).numpy()

    output_smi = output_smi + output2smiles(
        pred, eos_idx=token2idx[Constants.EOS_WORD], idx2token=idx2token, drop_first=0)

output_mol = list(map(Chem.MolFromSmiles, output_smi))
output_mol_valid = list(filter(None, output_mol))
fraction_valid = len(output_mol_valid) / len(output_mol)

logger.info(f"Valid molecules: {len(output_mol_valid)}/{len(output_mol)} ({fraction_valid * 100:.2f} %)")
