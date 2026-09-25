# Analyzing Autoencoders

This project compares fully connected and convolutional autoencoders for reconstructing 28×28 EMNIST Balanced character images. The notebook trains shallow and deep ANN models alongside light and deep CNN models, evaluates reconstruction error and compression, visualizes latent spaces with PCA, and explores interpolation between encoded samples.

## Architecture and overview

The repository is centered on [`emnist-aims.ipynb`](./emnist-aims.ipynb). Its data flow is:

```text
EMNIST Balanced CSV files
        │
        ▼
EMNISTCSV Dataset → DataLoaders → four autoencoders
                                  ├─ ANN_AE_Shallow
                                  ├─ ANN_AE_Deep
                                  ├─ CNN_AE_Light
                                  └─ CNN_AE_Deep
                                        │
                                        ▼
                         MSE evaluation, reconstructions,
                         PCA latent projection, interpolation
```

An interactive repository architecture view is available at [GitDiagram](https://gitdiagram.com/Ritwik389/Analyzing-Autoencoders).

### Model designs

- **ANN Shallow:** flattened 784-pixel input, bottleneck size 64, batch normalization, and a symmetric fully connected decoder.
- **ANN Deep:** a deeper fully connected encoder/decoder with a default bottleneck size of 32 and dropout layers.
- **CNN Light:** strided convolutions reduce 28×28 inputs to 7×7 feature maps, followed by a 128-dimensional bottleneck.
- **CNN Deep:** a larger three-stage convolutional encoder and transposed-convolution decoder with a 128-dimensional bottleneck.

## Key features

- Custom `EMNISTCSV` dataset loader for Kaggle EMNIST CSV exports.
- Reproducible train/validation split and seeded NumPy/PyTorch initialization.
- Adam optimization, MSE reconstruction loss, cosine-annealing learning-rate scheduling, and best-validation checkpoint restoration.
- Side-by-side reconstruction and training/validation-loss visualizations.
- PCA projections of latent representations colored by character class.
- Latent-space interpolation and learned convolution-filter visualizations.
- Saved example figures in [`Outputs/`](./Outputs).

## Tech stack

- Python 3
- PyTorch and Torchvision
- NumPy and Pandas
- Matplotlib
- scikit-learn (`PCA`)
- Jupyter Notebook

## Setup and installation

The notebook currently uses the Kaggle EMNIST dataset path
`/kaggle/input/datasets/crawford/emnist/` and is therefore easiest to run in Kaggle.

1. Install the dependencies in a Python 3 environment:

   ```bash
   python -m pip install jupyter numpy pandas matplotlib scikit-learn torch torchvision
   ```

2. Make the EMNIST Balanced CSV files available at the path configured in `CFG["data_root"]`. The notebook expects:

   - `emnist-balanced-train.csv`
   - `emnist-balanced-test.csv`
   - `emnist-balanced-mapping.txt`

3. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook emnist-aims.ipynb
   ```

   For local execution, change `CFG["data_root"]` to the directory containing those files. A CUDA-capable GPU is optional; the notebook automatically selects CUDA when available and otherwise uses the CPU.

## Usage

Run the notebook from top to bottom. The main configuration is near the first code cell:

```python
CFG.update({
    "split": "balanced",
    "batch_size": 128,
    "epochs": 30,
    "lr": 1e-3,
    "val_frac": 0.1,
})
```

After training, evaluate a model and inspect its bottleneck representation:

```python
model = trained_models["CNN_Deep"]
test_mse = evaluate(model, test_loader, nn.MSELoss(), CFG["device"])
print(test_mse, model.bottleneck)
```

The notebook produces the comparison figures shown in [`Outputs/`](./Outputs), including reconstruction comparisons, loss curves, model comparisons, latent-space projections, and latent interpolations.

## Status

**In Progress.** The notebook contains a working exploratory training and visualization workflow, but the repository is not yet a complete, reproducible software package.

Concrete remaining work identified during the audit:

- Extract the notebook implementation into importable Python modules or a documented notebook pipeline; there are currently no `.py` entry points.
- Add a dependency lock/manifest such as `requirements.txt` or `pyproject.toml`; dependencies are currently only implicit in notebook imports.
- Remove the hard-coded Kaggle path in `CFG["data_root"]` or replace it with a documented command-line/environment configuration.
- Add automated tests for `EMNISTCSV`, model output shapes, training/evaluation helpers, and latent interpolation; no test files are present.
- Add CI/build configuration; no workflow or other build configuration is present.
- Resolve the notebook’s open research questions and planned experiments, including trying other schedulers, improving ANN architectures, finding a smaller optimal bottleneck, and investigating class separation and skip-connection interpolation.

There are no `TODO` or `FIXME` markers in the tracked source, and no stub functions were found. The repository has no license file, so no license is declared.

## License

No license.
