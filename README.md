# GNN AI Code Detector

AI-generated code detection for C and C++ using Clang ASTs and a Relational Graph Convolutional Network (R-GCN).

## Overview

**GNN AI Code Detector** is a machine learning project that classifies C and C++ source code as either **human-written** or **AI-generated**.

Instead of processing source code as raw text, the project parses it with Clang and converts the resulting Abstract Syntax Tree (AST) into a typed graph. A Relational Graph Convolutional Network then learns representations of these graphs and performs graph-level classification.

The goal is to investigate whether structural properties of source code can provide useful signals for distinguishing human-written and AI-generated programs.

## Features

* **AST-based representation** — parses C/C++ source code using Clang rather than treating it as raw text.
* **Graph representation** — converts ASTs into graphs containing both structural and declaration/reference relationships.
* **Typed graph relations** — models `CHILD`, `PARENT`, `REFERENCE`, and `USAGE` relationships separately.
* **Identifier-independent features** — identifier names are not used as model features, encouraging the model to rely on structural properties rather than naming patterns.
* **Learned AST embeddings** — learns embeddings for categorical features such as AST node kinds, operation codes, and cast types.
* **Relational GCN** — uses two `RGCNConv` layers to propagate information across different relation types.
* **Graph-level classification** — combines global mean and max pooling to represent an entire source file.
* **C and C++ support** — currently supports both languages through Clang.
* **Single-file inference** — classify individual C/C++ source files from the command line.
* **Pretrained checkpoints** — trained model checkpoints are included in the repository.
* **GPU acceleration** — automatically uses an available XPU or CUDA device, falling back to CPU.

## How It Works

The overall pipeline is:

```text
C/C++ source
     │
     ▼
   Clang
     │
     ▼
  AST (JSON)
     │
     ├── Prune compiler-generated/external nodes
     ├── Remove irrelevant metadata
     └── Remove selected AST node types
     │
     ▼
 AST-derived graph
     │
     ├── CHILD / PARENT edges
     └── REFERENCE / USAGE edges
     │
     ▼
 PyTorch Geometric graph
     │
     ▼
    R-GCN
     │
     ├── Node feature embeddings
     ├── 2 × RGCNConv
     ├── Global mean pooling
     └── Global max pooling
     │
     ▼
 Classification MLP
     │
     ▼
 Human / AI
```

### AST preprocessing

Clang is used to generate a JSON representation of the C/C++ AST. The preprocessing pipeline removes information that is not intended to contribute to classification, including compiler-generated nodes, external include information, macro-expansion metadata, and selected AST node types.

The resulting graph does not use identifier names as node features.

### Graph representation

Each AST node becomes a graph node. Four directed relation types are constructed:

| Relation    | Description                      |
| ----------- | -------------------------------- |
| `CHILD`     | Parent AST node → child AST node |
| `PARENT`    | Child AST node → parent AST node |
| `REFERENCE` | Node → declaration it references |
| `USAGE`     | Declaration → node using it      |

The model currently uses the following node features:

| Feature    | Type        | Description                       |
| ---------- | ----------- | --------------------------------- |
| `kind`     | Categorical | Clang AST node type               |
| `opcode`   | Categorical | Operation code where applicable   |
| `castKind` | Categorical | Cast type where applicable        |
| `isArrow`  | Boolean     | Whether a member access uses `->` |

Categorical features are mapped to learned embeddings before being passed to the graph network.

## Model

The classifier is a **Relational Graph Convolutional Network (R-GCN)** implemented using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

The architecture is:

1. Learn embeddings for categorical AST features:

   * `kind`: 32 dimensions
   * `opcode`: 16 dimensions
   * `castKind`: 8 dimensions
2. Concatenate the embeddings with boolean features.
3. Project the resulting feature vector to 128 dimensions.
4. Apply an `RGCNConv` layer with four relation types.
5. Apply ReLU and dropout.
6. Apply a second `RGCNConv` layer.
7. Apply ReLU.
8. Perform global mean pooling and global max pooling.
9. Concatenate the two graph-level representations.
10. Pass the result through a two-layer MLP.
11. Produce two classification logits: **Human** and **AI**.

The model therefore performs **graph-level classification**, with each source file represented by a single graph.

## Dataset

The project uses a dataset containing **10,000 annotated code samples**:

The dataset is balanced at the overall level, with 5,000 human-written and 5,000 AI-generated samples. However, the distribution between the two classes varies by programming language.

This project currently restricts the dataset to **C and C++**, giving it a total of **4,377 samples** before the project-specific AST availability filtering:

* 968 AI-generated C samples
* 769 human-written C samples
* 1,221 AI-generated C++ samples
* 1,419 human-written C++ samples

### Dataset splitting

Samples are split by `problem_id` rather than by individual source file. This prevents samples belonging to the same programming problem from appearing in different dataset splits.

The data is divided into:

* **60% training**
* **20% validation**
* **20% test**

The project also excludes samples for which a corresponding AST is not available.

Feature vocabularies are constructed from the training set only, preventing validation and test samples from influencing the learned categorical feature mappings.

## Training

Training uses:

* **Optimizer:** Adam
* **Learning rate:** `1e-3`
* **Weight decay:** `1e-4`
* **Loss:** Cross-entropy
* **Batch size:** 32
* **Maximum epochs:** 30
* **Early stopping patience:** 6 epochs
* **Random seed:** 42
* **Hidden dimension:** 128

The checkpoint with the highest validation accuracy is saved and subsequently used for final test evaluation.

Training can be started with:

```bash
python -m gnn_ai_code_detector.train C
```

or for both supported languages:

```bash
python -m gnn_ai_code_detector.train C C++
```

The training script automatically selects an available XPU or CUDA device and falls back to CPU when no accelerator is available.

## Installation

Clone the repository and install it as a Python package:

```bash
git clone https://github.com/silazemli/gnn-ai-code-detector.git
cd gnn-ai-code-detector

python -m venv .venv
```

Activate the virtual environment and install the project:

```bash
pip install -e .
```

The project currently targets **Python 3.13**.

### Clang

Clang is required to parse C/C++ source code and generate ASTs.

The current development environment uses LLVM/Clang on Windows together with MSYS2-provided C/C++ libraries. The Clang executable and library paths are currently configured in the source code, so these paths may need to be adjusted for another environment.

A GPU is not required. PyTorch automatically uses an available accelerator when supported.

## Inference

Pretrained model checkpoints are included in the `models/` directory.

To classify a source file:

```bash
python -m gnn_ai_code_detector.inference path/to/source.cpp
```

The language is inferred from the file extension:

| Extension | Language |
| --------- | -------- |
| `.c`      | C        |
| `.cpp`    | C++      |
| `.cc`     | C++      |
| `.cxx`    | C++      |

The corresponding pretrained checkpoint is loaded automatically.

For example:

```text
Prediction: AI
Human:     0.000
AI:        1.000
```

## Repository Structure

```text
.
├── data/
│   ├── Code_Dataset/
│   └── c_cpp/
├── models/
│   ├── C_best_model.pt
│   └── C++_best_model.pt
├── notebooks/
│   └── c_cpp/
├── scripts/
│   └── c_cpp/
│       ├── build_asts.py
│       ├── clean_asts.py
│       └── prune_asts.py
├── src/
│   └── gnn_ai_code_detector/
│       ├── dataset.py
│       ├── inference.py
│       ├── model.py
│       ├── preprocess.py
│       ├── split.py
│       └── train.py
├── pyproject.toml
└── requirements.txt
```

### Main components

* `preprocess.py` — Clang AST preprocessing and graph construction
* `model.py` — R-GCN model definition
* `dataset.py` — PyTorch Geometric dataset handling
* `split.py` — problem-level dataset splitting
* `train.py` — model training and evaluation
* `inference.py` — single-file inference
* `scripts/c_cpp/` — AST generation and preprocessing utilities
* `notebooks/` — exploratory analysis and evaluation

## Results

The current model achieves the following results on the held-out test set:

| Language |  Accuracy | Precision | Recall |        F1 | AUROC | Average Precision |
| -------- | --------: | --------: | -----: | --------: | ----: | ----------------: |
| C        |     0.960 |     0.979 |  0.941 |     0.960 | 0.994 |             0.994 |
| C++      |     0.989 |     0.986 |  0.990 |     0.988 | 0.999 |             0.999 |
| C/C++    |     0.973 |     0.953 |  0.988 |     0.970 | 0.998 |             0.997 |

### Normalized confusion matrices

#### C

|           | Predicted Human | Predicted AI |
| --------- | --------------: | -----------: |
| **Human** |           0.980 |        0.020 |
| **AI**    |           0.059 |        0.941 |

#### C++

|           | Predicted Human | Predicted AI |
| --------- | --------------: | -----------: |
| **Human** |           0.988 |        0.012 |
| **AI**    |           0.010 |        0.990 |

#### C/C++

|           | Predicted Human | Predicted AI |
| --------- | --------------: | -----------: |
| **Human** |           0.961 |        0.039 |
| **AI**    |           0.012 |        0.988 |

These results are measured on held-out problems from the same dataset distribution. 

## Limitations

* The detector currently supports only C and C++.
* Evaluation is currently based on a single dataset.
* No baseline model comparison has been performed yet.
* Very short programs may contain little structural information from which authorship can be inferred.
* The current Clang/MSYS2 configuration is environment-dependent and may require modification on other systems.
* High performance on the current benchmark does not necessarily imply equivalent performance on code from different distributions.
* Standalone source files required — inference currently expects a self-contained source file that Clang can parse independently. Projects requiring external project files, custom include paths, generated headers, or additional compilation configuration may (will) fail during AST generation.

## Future Work

Potential areas for improvement include:

* Adding support for additional programming languages.
* Evaluating on larger and more diverse datasets.
* Performing cross-dataset evaluation to measure distribution-shift robustness.
* Adding baseline models for comparison.
* Testing generalization to previously unseen code-generation models.
* Investigating performance on short and structurally simple programs.
* Improving portability of the Clang and system-library configuration.
* Adding more detailed model and graph visualizations.

## References

The dataset used by this project is described in:

> *[A dataset for human-written and AI-generated code source classification]* — *Data in Brief*, 2026.