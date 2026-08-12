import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gnn_ai_code_detector.dataset import CCppDataset
from gnn_ai_code_detector.preprocess import CCppPreprocessor
from gnn_ai_code_detector.split import get_huvsai_split
from gnn_ai_code_detector.model import CCppGNN

from scripts.c_cpp.build_asts import build_asts
from scripts.c_cpp.cut_asts import cut_asts
from scripts.c_cpp.clean_asts import clean_asts

from pathlib import Path
import pandas as pd
import time

def train():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
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
def evaluate(loader):
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

if __name__ == "__main__":
    BUILD_ASTS = False
    CUT_ASTS   = False
    CLEAN_ASTS = False

    LANGUAGE = "C++"

    RANDOM_STATE = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_PATH = PROJECT_ROOT/"data"

    CSV_PATH = DATA_PATH/"Code_Dataset"/"HumanVsAi_CodeDataset.csv"

    RAW_AST_DIR   = DATA_PATH/"c_cpp"/"raw_asts"
    CUT_AST_DIR   = DATA_PATH/"c_cpp"/"cut_asts"
    CLEAN_AST_DIR = DATA_PATH/"c_cpp"/"clean_asts"

    CLANG_PATH = "C:/Program Files/LLVM/bin/clang.exe"
    preprocessor = CCppPreprocessor(CLANG_PATH)

    WORKERS = 12

    if BUILD_ASTS: build_asts(CSV_PATH,    RAW_AST_DIR,   preprocessor, True, WORKERS)
    if CUT_ASTS:     cut_asts(RAW_AST_DIR, CUT_AST_DIR,   preprocessor, True, WORKERS)
    if CLEAN_ASTS: clean_asts(CUT_AST_DIR, CLEAN_AST_DIR, preprocessor, True, WORKERS)

    df = pd.read_csv(CSV_PATH)
    train_indices, val_test_indices = get_huvsai_split(
        df, RAW_AST_DIR, LANGUAGE,
        test_size=0.4, random_state=RANDOM_STATE
    )

    test_indices, val_indices = get_huvsai_split(
        df.loc[val_test_indices], RAW_AST_DIR, LANGUAGE,
        test_size=0.5, random_state=RANDOM_STATE
    )

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")
    print(f"Test samples:  {len(test_indices)}")

    train_dataset = CCppDataset(train_indices, CLEAN_AST_DIR, CSV_PATH, preprocessor)
    val_dataset   = CCppDataset(val_indices,   CLEAN_AST_DIR, CSV_PATH, preprocessor)
    test_dataset  = CCppDataset(test_indices,  CLEAN_AST_DIR, CSV_PATH, preprocessor)

    print("Building vocabularies...")
    vocab = preprocessor.build_vocabularies(train_dataset)
    print("Done building vocabularies")

    train_dataset.vocab = vocab
    val_dataset.vocab   = vocab
    test_dataset.vocab  = vocab

    BATCH_SIZE = 32

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  BATCH_SIZE, shuffle=False)

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
    patience = 3
    epochs_without_improvement = 0

    EPOCHS = 20

    MODEL_PATH = "models/" + LANGUAGE + "_best_model.pt"

    for epoch in range(EPOCHS):
        start = time.time()

        loss, train_acc = train()

        val_acc = evaluate(val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)    
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("patience lost", end="")
            break

        print(
        f"{epoch+1:3d} | "
        f"loss: {loss:.2f} | "
        f"train / val: {train_acc:.3f} / {val_acc:.3f} | "
        f"et: {(time.time() - start):.0f}"
        f"{" | new best!" if epochs_without_improvement == 0 else ""}"
    )

    model.load_state_dict(torch.load(MODEL_PATH))
    test_acc = evaluate(test_loader)

    print(f"\ntest: {test_acc:.3f}")