from __future__ import annotations
import torch
from torch_geometric.data import Data
from pathlib import Path
import json
import subprocess
from enum import Enum

class Edge(Enum):
    CHILD = 0
    PARENT = 1
    REFERENCE = 2
    USAGE = 3

class ClangASTConverter:
    C_PREAMBLE = """
    #include <assert.h>
    #include <ctype.h>
    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    """

    CPP_PREAMBLE = """
    #include <algorithm>
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <cstdint>
    #include <iostream>
    #include <map>
    #include <set>
    #include <string>
    #include <unordered_map>
    #include <unordered_set>
    #include <vector>
    """

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
        "kind", "inner", "opcode",
        "isArrow", "castKind", "id",
        "referencedDecl",
        "referencedMemberDecl"
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

    NODE_FEATURES = {
        "kind", "opcode",
        "castKind", "isArrow"
    }

    def __init__(self, clang_path: str = "clang"):
        self.clang_path = clang_path

    def build_ast(self, path: Path) -> dict:
        result = subprocess.run(
            [self.clang_path, *self.CLANG_ARGS, str(path)],
            capture_output=True, text=True, check=True,
        )

        return json.loads(result.stdout)

    def append_headers(self, source: str, language: str) -> str:
        if language == "C":
            preamble = self.C_PREAMBLE
        elif language == "C++":
            preamble = self.CPP_PREAMBLE
        else: 
            raise(ValueError("Incorrect language specification, must be C/C++"))

        return preamble + "\n" + source
    
    def cut_irrelevant_branches(self, ast: dict) -> dict:
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
                edges["references"].append(reference_id)

            graph[node_id] = {"edges": edges, "features": features}

            for child in node.get("inner", []):
                child_id = child["id"]

                edges["children"].append(child_id)

                visit(child)

        visit(ast)

        return graph

    def handle_external_references(self, graph: dict) -> dict:
        node_ids = set(graph)

        for data in graph.values():
            data["edges"]["references"] = [
                reference
                for reference in data["edges"]["references"]
                if reference in node_ids
            ]

        return graph
                
    def build_vocabularies(self, ast_dir: Path) -> dict:
        kinds = set()
        opcodes = set()
        cast_kinds = set()

        def walk(node: dict):
            if isinstance(node, dict):
                if "kind" in node:
                    yield node

                for value in node.values():
                    yield from walk(value)

            elif isinstance(node, list):
                for item in node:
                    yield from walk(item)
                
        for path in ast_dir.glob("*.json"):
            with path.open("r", encoding="utf-8") as f:
                ast = json.load(f)

            for node in walk(ast):
                kinds.add(node["kind"])

                if "opcode" in node:
                    opcodes.add(node["opcode"])

                if "castKind" in node:
                    cast_kinds.add(node["castKind"])

        def make_vocab(values, special):
            vocab = {special: 0}
            for value in sorted(values):
                vocab[value] = len(vocab)
            return vocab

        return {
            "kind": make_vocab(kinds, "<UNK>"),
            "opcode": make_vocab(opcodes, "<NONE>"),
            "castKind": make_vocab(cast_kinds, "<NONE>"),
        }

    def construct_pyg_data(self, graph: dict, vocab: dict):
        node_to_idx = {node_id: idx for idx, node_id in enumerate(graph)}

        edges = []
        edge_types = []

        kinds = []
        opcodes = []
        cast_kinds = []
        is_arrows = []

        def add_edge(src, dst, edge_type: Edge):
            edges.append((src, dst))
            edge_types.append(edge_type.value)

        for node_id, node in graph.items():
            src = node_to_idx[node_id]
            features = node["features"]

            kinds.append(vocab["kind"].get(features["kind"], vocab["kind"]["<UNK>"]))
            opcodes.append(vocab["opcode"].get(features.get("opcode"), vocab["opcode"]["<NONE>"]))
            cast_kinds.append(vocab["castKind"].get(features.get("castKind"), vocab["castKind"]["<NONE>"]))
            is_arrows.append(features.get("isArrow", False))

            for child_id in node["edges"]["children"]:
                dst = node_to_idx[child_id]

                add_edge(src, dst, Edge.CHILD)
                add_edge(dst, src, Edge.PARENT)

            for reference_id in node["edges"]["references"]:
                dst = node_to_idx[reference_id]

                add_edge(src, dst, Edge.REFERENCE)
                add_edge(dst, src, Edge.USAGE)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)

        kind = torch.tensor(kinds, dtype=torch.long)
        opcode = torch.tensor(opcodes, dtype=torch.long)
        cast_kind = torch.tensor(cast_kinds, dtype=torch.long)
        is_arrow = torch.tensor(is_arrows, dtype=torch.bool)

        return Data(
            kind=kind, opcode=opcode,
            cast_kind=cast_kind, is_arrow=is_arrow,
            edge_index=edge_index, edge_type=edge_type,
        )