from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data

from gnn_ai_code_detector.train      import ModelCheckpoint
from gnn_ai_code_detector.model      import CCppGNN
from gnn_ai_code_detector.preprocess import CCppPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT/"models"

CLANG_PATH = "C:/Program Files/LLVM/bin/clang.exe"

def source_to_data(
        source: str,
        language: str,
        preprocessor: CCppPreprocessor,
        vocab: dict
    ):
    ast = preprocessor.build_ast(source, language)

    ast = preprocessor.prune(ast)
    ast = preprocessor.remove_metadata(ast)
    ast = preprocessor.remove_irrelevant_nodes(ast)

    graph = preprocessor.construct_graph(ast)

    data = preprocessor.construct_pyg_data(graph, vocab)

    return data

def predict(
        data: Data,
        model: nn.Module,
        device: torch.device
    ):
    model.eval()

    data = data.to(device)

    with torch.no_grad():
        logits = model(data)

        probabilities = torch.softmax(logits, dim=-1)[0]

    prediction = probabilities.argmax().item()
            
    return prediction, probabilities

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python -m gnn_ai_code_detector.inference <source-file>"
        )
        raise SystemExit(1)
        
    source_path = Path(sys.argv[1])

    if not source_path.is_file():
        print(f"File not found: {source_path}")
        raise SystemExit(1)

    suffix = source_path.suffix.lower()

    if suffix == ".c":
        language = "C"
    elif suffix in {".cpp", ".cc", ".cxx"}:
        language = "C++"
    else:
        print(f"Unsupported file extension: {source_path.suffix}")
        raise SystemExit(1)

    checkpoint_path = MODEL_DIR/f"{language}_best_model.pt"

    checkpoint = ModelCheckpoint.load(checkpoint_path)

    preprocessor = CCppPreprocessor(CLANG_PATH)

    model = CCppGNN(
        checkpoint.vocab,
        checkpoint.embedding_dims,
        checkpoint.num_relations,
        checkpoint.bool_features,
    )

    model.load_state_dict(checkpoint.state_dict)

    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    source = source_path.read_text(encoding="utf-8")

    data = source_to_data(
        source, language,
        preprocessor,
        checkpoint.vocab,
    )

    prediction, probabilities = predict(data, model, device)

    labels = ["Human", "AI"]

    print(f"Prediction: {labels[prediction]}")
    print(f"Human:     {probabilities[0].item():.3f}")
    print(f"AI:        {probabilities[1].item():.3f}")