from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gnn_ai_code_detector.dataset    import CCppDataset
from gnn_ai_code_detector.preprocess import CCppPreprocessor
from gnn_ai_code_detector.split      import get_huvsai_split
from gnn_ai_code_detector.model      import CCppGNN

import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import random
import time

@dataclass
class ModelCheckpoint:
    state_dict: dict
    vocab: dict
    embedding_dims: dict
    num_relations: int
    bool_features: list[str]
    languages: list[str]
    best_val_acc: float

    def save(self, path: Path):
        torch.save(self, path)

    @classmethod
    def load(cls, path: Path) -> ModelCheckpoint:
        return torch.load(path, weights_only=False)

def train(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        logits = model(data)

        loss = criterion(logits, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()*data.num_graphs

        predictions = logits.argmax(dim=1)
        correct += (predictions == data.y).sum().item()
        total += data.num_graphs

    loss = total_loss / total
    accuracy = correct / total

    return loss, accuracy

@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device
    ):
    model.eval()

    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)

        logits = model(data)

        predictions = logits.argmax(dim=1)

        correct += (predictions == data.y).sum().item()

        total += data.num_graphs

    accuracy = correct / total

    return accuracy

@torch.no_grad()
def predict(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device
    ):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for data in loader:
        data = data.to(device)

        logits = model(data)
        probabilities = torch.softmax(logits, dim=1)

        y_true.extend(data.y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_prob.extend(probabilities[:, 1].cpu().tolist())

    return y_true, y_pred, y_prob

if __name__ == "__main__":
    BUILD_ASTS = False
    CUT_ASTS = False
    CLEAN_ASTS = False

    LANGUAGES = sorted(sys.argv[1:])
    LANGUAGES = sorted(LANGUAGES) # so that there is no C/C++ C++/C distinction

    RANDOM_STATE = 42

    random.seed(RANDOM_STATE), np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_STATE)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_PATH = PROJECT_ROOT/"data"

    CSV_PATH = DATA_PATH/"Code_Dataset"/"HumanVsAi_CodeDataset.csv"

    AST_DIR = DATA_PATH/"c_cpp"
    RAW_AST_DIR = AST_DIR/"raw_asts"
    CUT_AST_DIR = AST_DIR/"cut_asts"
    CLEAN_AST_DIR = AST_DIR/"clean_asts"

    CLANG_PATH = "C:/Program Files/LLVM/bin/clang.exe"
    preprocessor = CCppPreprocessor(CLANG_PATH)

    WORKERS = 12

    if BUILD_ASTS:
        from scripts.c_cpp.build_asts import build_asts
        build_asts(CSV_PATH, RAW_AST_DIR, preprocessor, True, WORKERS)
    if CUT_ASTS:
        from scripts.c_cpp.prune_asts import prune_asts
        prune_asts(RAW_AST_DIR, CUT_AST_DIR, preprocessor, True, WORKERS)
    if CLEAN_ASTS:
        from scripts.c_cpp.clean_asts import clean_asts
        clean_asts(CUT_AST_DIR, CLEAN_AST_DIR, preprocessor, True, WORKERS)

    dataset = pd.read_csv(CSV_PATH)
    train_indices, val_test_indices = get_huvsai_split(
        dataset, RAW_AST_DIR, LANGUAGES,
        test_size=0.4, random_state=RANDOM_STATE
    )

    test_indices, val_indices = get_huvsai_split(
        dataset.loc[val_test_indices],
        RAW_AST_DIR, LANGUAGES,
        test_size=0.5, random_state=RANDOM_STATE
    )

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")
    print(f"Test samples:  {len(test_indices)}")

    train_dataset = CCppDataset(train_indices, CLEAN_AST_DIR, dataset, preprocessor)
    val_dataset = CCppDataset(val_indices, CLEAN_AST_DIR, dataset, preprocessor)
    test_dataset = CCppDataset(test_indices, CLEAN_AST_DIR, dataset, preprocessor)

    print("Building vocabularies...")
    vocab = preprocessor.build_vocabularies(train_dataset)
    print("Done building vocabularies")

    train_dataset.vocab = vocab
    val_dataset.vocab = vocab
    test_dataset.vocab = vocab

    BATCH_SIZE = 32

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    if torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    device = torch.device(device)

    EMBEDDING_DIMS = {
        "kind": 32,
        "opcode": 16,
        "castKind": 8
    }

    from gnn_ai_code_detector.preprocess import Edge
    NUM_RELATIONS = len(Edge)

    model = CCppGNN(
        vocab, EMBEDDING_DIMS, NUM_RELATIONS,
        preprocessor.BOOL_FEATURES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs_without_improvement = 0

    EPOCHS = 30
    PATIENCE = 6

    MODEL_PATH = "models/" + "_".join(LANGUAGES) + "_best_model.pt"

    print(f"Training for {EPOCHS} epochs with {PATIENCE}-epoch patience on the {", ".join(LANGUAGES)} data subset")
    for epoch in range(EPOCHS):
        start = time.time()

        loss, train_acc = train(
            model, train_loader, optimizer,
            criterion, device
        )

        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            checkpoint = ModelCheckpoint(
                model.state_dict(), vocab, EMBEDDING_DIMS,
                NUM_RELATIONS, preprocessor.BOOL_FEATURES,
                LANGUAGES, best_acc
            )
            checkpoint.save(MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("patience lost", end="")
            break

        print(
        f"{epoch+1:3d} | "
        f"loss: {loss:.2f} | "
        f"train/val: {train_acc:.3f}/{val_acc:.3f} | "
        f"et: {(time.time() - start):.0f}"
        f"{" | new best!" if epochs_without_improvement == 0 else ""}"
    )

    checkpoint = ModelCheckpoint.load(MODEL_PATH)

    model = CCppGNN(
        checkpoint.vocab,
        checkpoint.embedding_dims,
        checkpoint.num_relations,
        checkpoint.bool_features,
    ).to(device)

    model.load_state_dict(checkpoint.state_dict)

    test_acc = evaluate(model, test_loader, device)

    print(f"\ntest: {test_acc:.3f}")