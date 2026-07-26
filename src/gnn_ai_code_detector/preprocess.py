from gnn_ai_code_detector.meta_ast import NodeKind, MetaNode, MetaAST
from pathlib import Path
import json
import subprocess

class ClangMetaConverter:
    def __init__(self, mapping: dict[str, NodeKind], clang_path: str = "clang"):
        self.mapping = mapping
        self.clang_path = clang_path

    def convert(self, path: Path) -> MetaAST:
        clang_ast = self._build_ast(path)

        branches = self._cut_relevant_branches(clang_ast)

    def _build_ast(self, path: Path) -> dict:
        result = subprocess.run(
            [
                self.clang_path,
                "-Xclang", "-ast-dump=json",
                "-fsyntax-only",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout)

    def _cut_relevant_branches(self, ast: dict, path: Path) -> list[dict]:
        # Removes the includes
        source_file = str(path)

        def is_relevant(node: dict) -> bool:
            return (
                node.get("loc", {}).get("file") == source_file or
                node.get("name") == "main"
            )

        return [
            child
            for child in ast.get("inner", [])
            if is_relevant(child)
        ]

    def _convert_node(self, node: dict) -> MetaNode:
        pass