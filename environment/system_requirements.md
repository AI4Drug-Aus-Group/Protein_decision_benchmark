# System requirements

## Core analyses

- Python 3.10 or later.
- Dependencies listed in `requirements.txt` or `environment.yml`.
- Bash, curl and unzip for the ProteinGym download script.
- Storage for the downloaded archives, extracted score tables and generated results.

The environment files specify minimum dependency versions. HDF5 embedding analyses additionally require `h5py`.

## Parallel execution

Scripts exposing `--workers` distribute independent assay computations across processes. Set this option to match the available CPU cores and memory. The zero-shot evaluation also provides `--method-chunk-size` to control the number of predictor columns processed together.

## External models

ProteinNPT and Kermut require their published implementations, model resources and software environments. Neural-model and embedding runs use the PyTorch and accelerator settings of those implementations. Cached-embedding regression and acquisition analyses use the corresponding HDF5 files.

## Plot outputs

Scripts that generate plots use Matplotlib. The included PNG overview figures can be viewed directly.
