Processed ChEMBL30 chemreps

1. Ran through ProcessLibrary.py with
     python ProcessLibrary.py -i chembl_30_chemreps.csv -o chembl_30_chemreps_proc.smi -s SMILES -id name -o_sep ' ' --chunk_size 10000 --max_len 80
        That is, dropping SMILES > 80 chars, desalting, neutralising, canonicalising
2. Deduplicated by SMILES
3. Keep SMILES that only contain tokens with more than 1000 occurrences across the corpus
3. train_test_split with sklearn (0.95 train, 0.025 valid, 0.025 test)
