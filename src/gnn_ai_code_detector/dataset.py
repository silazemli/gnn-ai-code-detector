import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pathlib import Path
import json
import pandas as pd

from gnn_ai_code_detector.preprocess import CCppPreprocessor

class CCppDataset(Dataset):
    def __init__(
            self,
            indices: list[int],
            ast_dir: Path,
            dataset: pd.DataFrame,
            preprocessor: CCppPreprocessor,
            vocab: dict | None = None
            ):
        self.indices = indices
        self.ast_dir = ast_dir
        self.labels = {
            index: 0 if generated == "Human" else 1
            for index, generated
            in dataset["Generated"].items()
        }
        self.preprocessor = preprocessor
        self.vocab = vocab

    def ast_paths(self):
        for idx in self.indices:
            yield self.ast_dir/f"{idx}.json"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        row_idx = self.indices[idx]

        path = self.ast_dir/f"{row_idx}.json"

        with path.open("r", encoding="utf-8") as f:
            ast = json.load(f)

        graph = self.preprocessor.construct_graph(ast)

        if self.vocab is None:
            raise RuntimeError("Vocabulary missing.")

        pyg_data = self.preprocessor.construct_pyg_data(graph, self.vocab)

        pyg_data.y = torch.tensor(self.labels[row_idx], dtype=torch.long)

        return pyg_data