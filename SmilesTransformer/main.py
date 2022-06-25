from numbers import Number
from typing import Optional

import torch.utils.data
from rdkit import Chem, RDLogger

from SmilesTransformer.loaders import build_loader
from SmilesTransformer.model.trainer import train
from SmilesTransformer.model.transformer import Constants
from SmilesTransformer.model.transformer.Models import Transformer
from SmilesTransformer.model.transformer.Optim import ScheduledOptim
from SmilesTransformer.utils import logger
from SmilesTransformer.utils import output2smiles

RDLogger.DisableLog("rdApp.*")


def main(
        model_config_path: str,
        train_csv_path: str,
        val_csv_path: str,
        src_SMILES_col: str,
        tgt_SMILES_col: str,
        alphabet_path: str,
        sample_train: Optional[Number],
        sample_val: Optional[Number],
        n_epochs: int = 1,
        label_smoothing: bool = False,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        augment_times: int = 0,
        num_workers: int = 1,
        random_state: Optional[int] = None,
        checkpoint_folder: str = './',
        csv_sep: str = ',', alphabet_sep='\n'
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device: {device}")

    logger.info("Creating training loader")
    train_loader, token2idx, idx2token = build_loader(
        csv_path=train_csv_path,
        src_col=src_SMILES_col, tgt_col=tgt_SMILES_col, alphabet_path=alphabet_path,
        sample=sample_train, random_state=random_state, batch_size=train_batch_size,
        num_workers=num_workers, augment_times=augment_times,
        csv_sep=csv_sep, alphabet_sep=alphabet_sep
    )

    logger.info("Creating validation loader")
    val_loader, _, _ = build_loader(
        csv_path=val_csv_path,
        src_col=src_SMILES_col, tgt_col=tgt_SMILES_col, alphabet_path=alphabet_path,
        sample=sample_val, random_state=random_state, batch_size=val_batch_size,
        num_workers=num_workers, augment_times=augment_times,
        csv_sep=csv_sep, alphabet_sep=alphabet_sep
    )

    logger.info("Instantiating the transformer")
    transformer = Transformer.from_config(model_config_path).to(device)

    optimizer = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        512, 4000)

    logger.info("Training")
    train(
        model=transformer, n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader, optimizer=optimizer,
        device=device,
        label_smoothing=label_smoothing,
        checkpoint_folder=checkpoint_folder
    )
    logger.info("Training finished")

    logger.info("Converting validation output to SMILES")
    transformer.eval()
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Smiles Transformer')
    parser.add_argument('-c', '--config', help='Transformer JSON config file', required=True)
    parser.add_argument('--train_path', help='Train SMILES CSV file', required=True)
    parser.add_argument('--val_path', help='Validation SMILES CSV file', required=True)
    parser.add_argument('--train_batch_size', help='Train batch size (default %(default)d)', default=64, type=int)
    parser.add_argument('--val_batch_size', help='Validation batch size (default %(default)d)', default=64, type=int)
    parser.add_argument('--src_smiles_col', help='Source SMILES column name (default: %(default)s)', default='SMILES')
    parser.add_argument('--tgt_smiles_col', help='Target SMILES column name (default: %(default)s)', default='SMILES')
    parser.add_argument('--alphabet_path', help='Alphabet file path', required=True)
    parser.add_argument('--num_epochs', help='Number of epochs (default %(default)s)', default=1, type=int)
    parser.add_argument('--sample_train', help='Randomly sample N pairs from training set (default: %(default)s)',
                        default=None, type=int)
    parser.add_argument('--sample_val', help='Randomly sample N pairs from validation set (default: %(default)s)',
                        default=None, type=int)
    parser.add_argument('--augment', help='Augment pairs N times with non-canonical SMILES (default: %(default)d)',
                        default=0, type=int)
    parser.add_argument('--num_workers', help='Number of workers for the DataLoaders (default: %(default)d)',
                        default=1, type=int)
    parser.add_argument('--random_state',
                        help='Random state for sampling training and validation set (default: %(default)d)', default=42,
                        type=int)
    parser.add_argument('--checkpoint_folder',
                        help='Folder where to save the model checkpoint (default: %(default)s)', default='./')
    parser.add_argument('--csv_sep', help='CSV column separator (default: %(default)s)', default=',')
    parser.add_argument('--alphabet_sep', help='Alphabet token separator (default: newline)', default='\n')
    args = parser.parse_args()

    main(
        model_config_path=args.config,
        train_csv_path=args.train_path, val_csv_path=args.val_path,
        train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
        src_SMILES_col=args.src_smiles_col, tgt_SMILES_col=args.tgt_smiles_col,
        alphabet_path=args.alphabet_path, n_epochs=args.num_epochs,
        sample_train=args.sample_train, sample_val=args.sample_val,
        augment_times=args.augment, random_state=args.random_state,
        num_workers=args.num_workers, checkpoint_folder=args.checkpoint_folder,
        csv_sep=args.csv_sep, alphabet_sep=args.alphabet_sep
    )
