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
def evaluate():
    model.eval()

    correct = 0
    total = 0

    for data in test_loader:
        data = data.to(device)

        logits = model(data)

        predictions = logits.argmax(dim=1)

        correct += (predictions == data.y).sum().item()

        total += data.num_graphs

    accuracy = correct / total

    return accuracy

if __name__ == "__main__":
    BUILD_ASTS = False
    CUT_ASTS = False
    CLEAN_ASTS = False

    LANGUAGE = "C/C++"

    RANDOM_STATE = 42

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_PATH = PROJECT_ROOT/"data"

    CSV_PATH = DATA_PATH/"Code_Dataset"/"HumanVsAi_CodeDataset.csv"

    RAW_AST_DIR = DATA_PATH/"c_cpp"/"raw_asts"
    CUT_AST_DIR = DATA_PATH/"c_cpp"/"cut_asts"
    CLEAN_AST_DIR = DATA_PATH/"c_cpp"/"clean_asts"

    CLANG_PATH = "C:/Program Files/LLVM/bin/clang.exe"
    preprocessor = CCppPreprocessor(CLANG_PATH)

    FORCE_BUILD = False
    FORCE_CUT = True
    FORCE_CLEAN = True

    WORKERS = 12

    if BUILD_ASTS: build_asts(CSV_PATH, RAW_AST_DIR, preprocessor, FORCE_BUILD, 4)
    if CUT_ASTS: cut_asts(RAW_AST_DIR, CUT_AST_DIR, preprocessor, FORCE_CUT, WORKERS)
    if CLEAN_ASTS: clean_asts(CUT_AST_DIR, CLEAN_AST_DIR, preprocessor, FORCE_CLEAN, WORKERS)

    train_indices, test_indices = get_huvsai_split(CSV_PATH, RAW_AST_DIR, LANGUAGE, random_state=RANDOM_STATE)

    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples:  {len(test_indices)}")

    train_dataset = CCppDataset(
        train_indices, CLEAN_AST_DIR, CSV_PATH,
        preprocessor
    )

    test_dataset = CCppDataset(
        test_indices, CLEAN_AST_DIR, CSV_PATH,
        preprocessor
    )

    print("Building vocabularies...")
    vocab = preprocessor.build_vocabularies(train_dataset)
    print("Done building vocabularies")

    train_dataset.vocab = vocab
    test_dataset.vocab = vocab

    BATCH_SIZE = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

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

    BOOL_FEATURES = ["isArrow"]

    from gnn_ai_code_detector.preprocess import Edge
    NUM_RELATIONS = len(Edge)

    model = CCppGNN(
        vocab, EMBEDDING_DIMS,
        NUM_RELATIONS, BOOL_FEATURES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 25
    epochs_without_improvement = 0

    EPOCHS = 20

    for epoch in range(EPOCHS):
        start = time.time()

        loss, train_acc = train()

        test_acc = evaluate()

        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/best_model.pt")    
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        print(
        f"{epoch+1:3d} | "
        f"loss: {loss:.2f} | "
        f"train / test: {train_acc:.3f} / {test_acc:.3f} | "
        f"et: {(time.time() - start):.0f}"
        f"{" | new best!" if epochs_without_improvement == 0 else ""}"
    )