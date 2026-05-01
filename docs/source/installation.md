# 📥 Installation

## ⚡️ Quick Start with GitHub Codespaces
The fastest way to start using the library is with **GitHub Codespaces**. It provides a pre-configured development environment in the cloud, so you can get started without setting up anything locally.

To use GitHub Codespaces:

  1. Click on&nbsp;: <br/> &nbsp;[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/p2pfl/p2pfl/tree/develop?quickstart=1)
  2. Create a new Codespace or open an existing one.
  3. Once the environment is ready, the library and its dependencies will already be set up.

Alternatively:

1. Navigate to the [p2pfl](https://github.com/p2pfl/p2pfl) repository.
2. Click on the green `<> Code` button, then select the codespaces tab.
3. Create a new Codespace or open an existing one.
4. Once the environment is ready, the library and its dependencies will already be set up.

## ⚙️ Manual Installation (Advanced)
If you prefer a fully manual installation or need to customize the setup, follow these steps:

### 👨🏼‍💻 For users

To install the library with specific dependencies, you can use one of the following commands:

- For PyTorch-related dependencies:

  ```bash
  pip install "p2pfl[torch]"
  ```

- For TensorFlow-related dependencies:

  ```bash
  pip install "p2pfl[tensorflow]"
  ```

If you want to install **all dependencies**, you can do so with:

```bash
pip install "p2pfl[torch,tensorflow]"
```

### 👨🏼‍🔧 For developers

#### 🐍 Python

> **Prerequisite**: Before installing the library, ensure that **UV** is installed. If you haven't installed it yet, follow the instructions in the official [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).


```bash
git clone https://github.com/pguijas/p2pfl.git
cd p2pfl
uv sync --all-extras
```

This installs p2pfl with all available dependencies (PyTorch and TensorFlow).

> **Note**: If you only need specific frameworks, you can use:
> - `uv sync` - Install only core p2pfl dependencies
> - `uv sync --extra torch` - Install with PyTorch support
> - `uv sync --extra tensorflow` - Install with TensorFlow support  
> 
> Additionally, you can use the `--no-dev` flag to install the library without development dependencies.


#### 🐳 Docker

You can also use the library with Docker. We provide a Docker image with the library installed. You can use it as follows:

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```


## 📦 Dependency Overview

This library supports **P2P Decentralized Federated Learning** with flexibility for different frameworks and backend integrations. You can choose dependencies tailored for your project’s needs. Below is an overview of the dependency options:

- **Torch**: For PyTorch-based deep learning models, including support for:

    - `torch`: Core PyTorch library.
    - `torchvision`: Tools for computer vision tasks.
    - `torchmetrics`: Metrics for evaluating models in PyTorch.
    - `lightning`: PyTorch Lightning, a framework for high-performance training.

- **TensorFlow**: For TensorFlow-based federated learning setups, including:

    - `tensorflow`: Core TensorFlow library.
    - `keras`: High-level API for building and training models in TensorFlow.
    - `types-tensorflow`: Type annotations for TensorFlow.

- **XGBoost**: For tree-based federated learning with gradient boosting.

- **Flax**: For JAX/Flax-based models.

- **DP**: For differential privacy support via OpenDP.

- **WandB**: For Weights & Biases experiment tracking.

