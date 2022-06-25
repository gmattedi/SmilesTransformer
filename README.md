# SMILES Transformer

### A SMILES-to-SMILES Transformer implementation

## Input data

Any set of SMILES or pairs of SMILES strings in CSV-like format can be used.  
A cleaned version of ChEMBL 30 is provided in [data/chembl_30](/data/chembl_30).

## Generating the token alphabet

The token alphabet can be generated from the ChEMBL training set with

```console
cd SmilesTransformer/tokenizer
python tokenizer.py
# Tokens are saved in alphabet.dat
```

A precomputed set can be found [here](/SmilesTransformer/tokenizer/alphabet.dat).

## Model configuration
The transformer can be instantiated from a JSON file (i.e. [config.json](config.json))
```json
{
  "n_src_vocab": 44,
  "n_tgt_vocab": 44,
  "len_max_seq": 100,
  "d_word_vec": 512,
  "d_model": 512,
  "d_inner": 2048,
  "n_layers": 6,
  "n_head": 8,
  "d_k": 64,
  "d_v": 64,
  "dropout": 0.1,
  "tgt_emb_prj_weight_sharing": true,
  "emb_src_tgt_weight_sharing": true
}
```
In the example provided, the alphabet length is that of the tokens of the ChEMBL 30 training set

## Training the model

You can train the model on a training and validation subsample of ChEMBL 30 of 1000 and 50 molecules, respectively, by:

```console
python SmilesTransformer/main.py \
    -c config.json \
    --train_path data/chembl_30/chembl_30_chemreps_proc_train.csv.gz \
    --val_path data/chembl_30/chembl_30_chemreps_proc_valid.csv.gz \
    --alphabet_path SmilesTransformer/tokenizer/alphabet.dat \
    --sample_train 1000 \
    --sample_val 50 \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --src_smiles_col SMILES \
    --tgt_smiles_col SMILES \
    --num_epochs 10 --augment 1 --checkpoint_folder .
```

In this case we are training the transformer to reconstruct the original SMILES strings,
but this can be trivially adapted to predicting to different target SMILES strings by
providing training and test CSV files of pairs of molecules.

## Credits

This repository uses the vanilla transformer implementation
by [siat-nlp](https://github.com/siat-nlp/transformer-pytorch).  
The SMILES tokenization regex pattern is from
the [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer).

## References

[1] Vaswani et al., [Attention Is All You Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), NIPS(
2017).

[2] A PyTorch
implementation [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).