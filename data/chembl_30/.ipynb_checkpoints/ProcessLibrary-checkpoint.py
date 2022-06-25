import multiprocessing as mp
import time
from typing import Sequence, List

import numpy as np
import pandas as pd
import tqdm as t
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

"""
Process library in a parallelised fashion

The molecules are desalted, neutralised and canonicalised
"""


def process_mols(mols: Sequence[Chem.Mol]) -> Sequence[Chem.Mol]:
    largest_Fragment = rdMolStandardize.LargestFragmentChooser()

    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
    )

    mols_processed = []

    for mol in mols:

        if mol is None:
            mols_processed.append(None)
            continue
        mol = largest_Fragment.choose(mol)
        if mol is None:
            mols_processed.append(None)
            continue

        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        success = True
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                try:
                    atom.UpdatePropertyCache()
                except Chem.AtomValenceException:
                    success = False
                if not success:
                    break

        if success:
            mols_processed.append(mol)
        else:
            mols_processed.append(None)

    return mols_processed


def process(smiles: Sequence[str]) -> List[str]:
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    mols = process_mols(mols)
    smiles = [Chem.MolToSmiles(mol) if mol is not None else None for mol in mols]
    return smiles


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Process library')
    parser.add_argument('-i', '--input_csv', help='Input CSV file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-s', '--smiles_col', help='Input SMILES columns (default: %(default)s)', required=False,
                        default='SMILES')
    parser.add_argument('-id', '--id_col', help='Input ID column', required=True)
    parser.add_argument(
        '-i_sep', '--input_sep', help='Input field separator (default: %(default)s)', required=False, default=',')
    parser.add_argument(
        '-o_sep', '--output_sep', help='Output field separator (default: %(default)s)', required=False, default=' ')
    parser.add_argument(
        '-header', '--output_header',
        help='Output field header string. Use the same separator as -sep (default: %(default)s)', required=False,
        default=None)
    parser.add_argument('--max_len', help='SMILES length threshold (default: %(default)d)', required=False,
                        default=9999, type=int)
    parser.add_argument('--chunk_size', help='Chunk size (default: %(default)d)', required=False, default=1000,
                        type=int)
    parser.add_argument('--n_cpu', help='CPU cores (default: %(default)d)', required=False, default=10, type=int)
    args = parser.parse_args()

    time_start = time.time()
    total_rows = 0

    pool = mp.Pool(args.n_cpu)

    if args.output_header is not None:
        fout = open(args.output, 'wt')
        fout.write(args.output_header + '\n')
        fout.close()
        fout = open(args.output, 'at')
    else:
        fout = open(args.output, 'wt')

    with pd.read_csv(args.input_csv, chunksize=args.chunk_size, sep=args.input_sep) as reader:
        for df in t.tqdm(reader, f'Processing file chunks ({args.chunk_size} lines)'):

            df = df[df[args.smiles_col].map(len) <= args.max_len]

            name_chunks, smiles_chunks = (
                np.array_split(df[args.id_col].values, args.n_cpu),
                np.array_split(df[args.smiles_col].values, args.n_cpu)
            )

            smiles_chunks_processed = pool.map(process, smiles_chunks)
            for name, smiles in zip(name_chunks, smiles_chunks_processed):
                smiles = np.array(smiles)
                mask = smiles != None

                batch = np.concatenate([
                    smiles[mask, np.newaxis],
                    name[mask, np.newaxis]
                ], axis=1)

                np.savetxt(fout, batch, fmt='%s', delimiter=args.output_sep)

            total_rows += df.shape[0]

    pool.close()
    pool.join()

    time_end = time.time()
    time_taken = time_end - time_start

    rows_per_second = total_rows / time_taken
    mins_per_mln_rows = 1e6 / rows_per_second / 60

    print(f'{total_rows} rows | {time_taken:.1f} s | {rows_per_second:.1f} rows/s | {mins_per_mln_rows:.1f} min/Mrows')
