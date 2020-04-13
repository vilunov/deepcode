# Token Preprocessing

Reads the dataset, tokenizes it using BPE algorithm and saves the resulting tokens into HDF5 file.

**Requirements:**
- Rust & cargo >=1.41
- hdf5 system package
  - `brew install hdf5`

## Binaries:

1. `preprocess-seq` – builds BPE vocabulary for docs and code, preprocess train data into HDF5 files
2. `preprocess-seq2` – preprocess test data into HDF5 file. Test data should be firstly preprocessed by a python script,
as we cannot deserialize pickle files here.