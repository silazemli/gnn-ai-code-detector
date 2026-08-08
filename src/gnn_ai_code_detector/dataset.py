from pathlib import Path
import json

from torch.utils.data import Dataset
from torch_geometric.data import Data

from gnn_ai_code_detector.preprocess import CCppPreprocessor

class CCppDataset(Dataset):
    def __init__(
            self,
            indices: list[int],
            ast_dir: Path,
            preprocessor: CCppPreprocessor,
            vocab: dict | None = None
            ):
        self.indices = indices
        self.ast_dir = ast_dir
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

        return pyg_data