from pathlib import Path
import json
import subprocess

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

    KEYS_TO_REMOVE = {
        "id", "loc", "range", "isUsed",
        "mangledName", "valueCategory"
    }

    COMPILER_ARTIFACTS = {
        "ImplicitCastExpr"
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
    
    def cut_irrelevant_branches(self, ast: dict, path: Path) -> dict:
        source_file = str(path)

        def is_relevant(node: dict) -> bool:
            return (
                node.get("loc", {}).get("file") == source_file or
                node.get("name") == "main"
            )

        cut_ast = ast.copy()

        cut_ast["inner"] = [
            child
            for child in ast.get("inner", [])
            if is_relevant(child)
        ]
        
        return cut_ast

    def remove_metadata(self, ast: dict) -> dict:
        def clean(value):
            if isinstance(value, dict):
                return {
                    key: clean(val)
                    for key, val in value.items()
                    if key not in self.KEYS_TO_REMOVE
                }
            elif isinstance(value, list):
                return [
                    clean(item)
                    for item in value
                ]
            else:
                return value

        return clean(ast)

    def remove_compiler_artifacts(self, ast: dict) -> dict:
        def clean(node):
            if isinstance(node, list):
                result = []
                for child in node:
                    cleaned = clean(child)

                    if cleaned is None:
                        continue

                    result.append(cleaned)

                return result

            if not isinstance(node, dict):
                return node

            if node.get("kind") in self.COMPILER_ARTIFACTS:
                inner = node.get("inner", [])
                return clean(inner[0]) if len(inner) == 1 else clean(inner)

            return {
                key: clean(value)
                for key, value in node.items()
            }

        return clean(ast)
    