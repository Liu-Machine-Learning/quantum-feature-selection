# Quantum Feature Selection
Code and datasets for our paper “Quantum Feature Selection for Biomedical Data Analysis”

## Recreate the QFS Python Environment
The environment is pinned to:

- Python 3.13.2
- Qiskit 2.4.0
- The exact package versions and Git commit revisions recorded in `uv.lock`

### Prerequisites

Install the following tools before continuing:

- [Git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 1. Get the repository

Clone the repository and enter its directory:

```bash
git clone <repository-url>
cd 
```

## 2. Install the required Python version

Install Python 3.13.2 through uv:

```bash
uv python install 3.13.2
```

## 3. Create and synchronize the environment

From the repository root, run:

```bash
uv sync --frozen
```

## 4. Activate the environment

```bash
source .venv/bin/activate
```

## 5. Verify the installation

Check the Python and Qiskit versions:

```bash
python --version
python -c "import qiskit; print(qiskit.__version__)"
```

Expected output:

```text
Python 3.13.2
2.4.0
```
