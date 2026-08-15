from pathlib import Path
import json
import subprocess
from enum import Enum

import torch
from torch_geometric.data import Data

class Edge(Enum):
    CHILD = 0
    PARENT = 1
    REFERENCE = 2
    USAGE = 3

class CCppPreprocessor:
    C_PREAMBLE = "".join(f"#include <{lib}.h>\n" for lib in [
        "assert", "ctype", "math", "stdio", "stdlib", "string"
    ])

    CPP_PREAMBLE = "".join(f"#include <{lib}>\n" for lib in [
        "algorithm", "cmath", "cstdio", "cstdlib", "cstring",
        "cstdint", "iostream", "map", "set", "string",
        "unordered_map", "unordered_set", "vector"
    ])

    CLANG_ARGS = [
        "--target=x86_64-w64-windows-gnu",
        "--gcc-install-dir=C:/msys64/ucrt64/lib/gcc/x86_64-w64-mingw32/16.1.0",

        "-w",
        "-DM_PI=3.14159265358979323846",

        "-isystem",
        "C:/msys64/ucrt64/include",

        "-isystem",
        "C:/msys64/ucrt64/include/opencv4",

        "-isystem",
        "C:/msys64/ucrt64/include/eigen3",

        "-isystem",
        "C:/msys64/ucrt64/include/cryptopp",

        "-Xclang",
        "-ast-dump=json",
        "-fsyntax-only",
    ]

    RELEVANT_METADATA = {
        "id", "referencedDecl", "referencedMemberDecl",
        "inner",
        "kind", "opcode", "isArrow", "castKind",
    }

    NODES_TO_REMOVE = {
        "ImplicitCastExpr",

        "UnusedAttr", "DLLImportAttr", "WarnUnusedResultAttr",
        "AlwaysInlineAttr", "DeprecatedAttr", "GNUInlineAttr",
        "NonNullAttr", "ReturnsNonNullAttr", "MaxFieldAlignmentAttr",
        "NoInlineAttr", "ErrorAttr", "OverrideAttr", "AlignedAttr",
        "PureAttr", "ConstAttr", "BuiltinAttr", "NoThrowAttr",

        "ParagraphComment", "TextComment", "FullComment",
        "BlockCommandComment",

        "TemplateArgument", "QualType", "DependentNameType",
        "DependentSizedArrayType", "InjectedClassNameType",
        "UnresolvedLookupExpr", "UnresolvedMemberExpr",
        "DependentScopeDeclRefExpr"
    }

    BOOL_FEATURES = ("isArrow", )

    EMBEDDED_FEATURES = {"kind", "opcode", "castKind"}

    NODE_FEATURES = set(BOOL_FEATURES) | EMBEDDED_FEATURES

    def __init__(self, clang_path: str = "clang"):
        self.clang_path = clang_path

    def build_ast(self, source: str, language: str) -> dict:
        source = self._append_headers(source, language)

        result = subprocess.run(
            [
                self.clang_path,
                *self.CLANG_ARGS,
                "-x", "c++" if language == "C++" else "c",
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def _append_headers(self, source: str, language: str) -> str:
        if language == "C":
            preamble = self.C_PREAMBLE
        elif language == "C++":
            preamble = self.CPP_PREAMBLE
        else: 
            raise(ValueError("Incorrect language specification, must be C/C++"))

        return preamble + "\n" + source
    
    def prune(self, ast: dict) -> dict:
        def irrelevant(node: dict) -> bool:
            return ( # remove
                # compiler-generated definitions
                node.get("isImplicit", False)
                # includes
                or node.get("loc", {}).get("includedFrom")
                # macro expansions from elsewhere
                or node.get("loc", {}).get("expansionLoc")
            )

        cut_ast = ast.copy()

        cut_ast["inner"] = [
            branch
            for branch in ast.get("inner", [])
            if not irrelevant(branch)
        ]
        
        return cut_ast

    def remove_metadata(self, ast: dict) -> dict:
        def clean(value):
            if isinstance(value, dict):
                return {
                    key: clean(val)
                    for key, val in value.items()
                    if key in self.RELEVANT_METADATA
                }
            elif isinstance(value, list):
                return [clean(item) for item in value]
            else:
                return value

        return clean(ast)

    def remove_irrelevant_nodes(self, ast: dict) -> dict:
        def clean(node):
            if "inner" not in node:
                return
            
            children = []

            for child in node["inner"]:
                if "id" not in child:
                    continue

                clean(child)

                if child.get("kind") in self.NODES_TO_REMOVE:
                    children.extend(child.get("inner", []))
                else:
                    children.append(child)

            node["inner"] = children
        
        clean(ast)

        return ast

    def construct_graph(self, ast: dict):
        graph: dict = {}
        
        def visit(node: dict):
            node_id = node["id"]

            edges = {"children": [], "references": []}

            features = {
                key: node[key]
                for key in self.NODE_FEATURES
                if key in node
            }

            referenced_decl: dict = node.get("referencedDecl", {})
            if referenced_decl:
                reference_id = referenced_decl.get("id", "")
                if reference_id: # is this check redundant? most likely
                    edges["references"].append(reference_id)

            referenced_member_decl: str = node.get("referencedMemberDecl", "")
            if referenced_member_decl:
                edges["references"].append(referenced_member_decl)

            graph[node_id] = {
                "edges": edges,
                "features": features
            }

            for child in node.get("inner", []):
                child_id = child["id"]
                edges["children"].append(child_id)
                visit(child)

        visit(ast)

        self._handle_external_references(graph)

        return graph

    def _handle_external_references(self, graph: dict):
        node_ids = set(graph)

        for data in graph.values():
            data["edges"]["references"] = [
                reference
                for reference in data["edges"]["references"]
                if reference in node_ids
            ]
                
    def build_vocabularies(self, dataset) -> dict:
        values = {feature: set() for feature in self.EMBEDDED_FEATURES}

        def walk(node: dict):
            if isinstance(node, dict):
                if "kind" in node:
                    yield node

                for value in node.values():
                    yield from walk(value)

            elif isinstance(node, list):
                for item in node:
                    yield from walk(item)
                
        for path in dataset.ast_paths():
            with path.open("r", encoding="utf-8") as f:
                ast = json.load(f)

            for node in walk(ast):
                for feature in self.EMBEDDED_FEATURES:
                    if feature in node:
                        values[feature].add(node[feature])

        return {
            feature: {
                value: i
                for i, value in enumerate(sorted(feature_values), start=1)
            } | {"<NONE>": 0} 
            for feature, feature_values in values.items()
        }

    def construct_pyg_data(self, graph: dict, vocab: dict) -> Data:
        node_to_idx = {node_id: idx for idx, node_id in enumerate(graph)}

        edges = []
        edge_types = []

        embedded = {feat: [] for feat in self.EMBEDDED_FEATURES}
        boolean =  {feat: [] for feat in self.BOOL_FEATURES}

        def add_edge(src, dst, edge_type: Edge):
            edges.append((src, dst))
            edge_types.append(edge_type.value)

        for node_id, node in graph.items():
            src = node_to_idx[node_id]
            feats = node["features"]

            for feat in self.EMBEDDED_FEATURES:
                    embedded[feat].append(vocab[feat].get(feats.get(feat), vocab[feat]["<NONE>"]))

            for feat in self.BOOL_FEATURES:
                boolean[feat].append(feats.get(feat, False))
            
            for child_id in node["edges"]["children"]:
                dst = node_to_idx[child_id]
                add_edge(src, dst, Edge.CHILD)
                add_edge(dst, src, Edge.PARENT)

            for reference_id in node["edges"]["references"]:
                dst = node_to_idx[reference_id]
                add_edge(src, dst, Edge.REFERENCE)
                add_edge(dst, src, Edge.USAGE)

        return Data(
            **{
                feat: torch.tensor(val, dtype=torch.long)
                for feat, val in embedded.items()
            },
            **{
                feat: torch.tensor(val, dtype=torch.bool)
                for feat, val in boolean.items()
            },
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            num_nodes=len(node_to_idx)
        )