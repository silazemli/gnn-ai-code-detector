from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AST_DIR = PROJECT_ROOT/"data"/"c_cpp"/"clean_asts"
OUTPUT_FILE = PROJECT_ROOT/"data"/"c_cpp"/"vocabularies.json"

def walk(node):
    if isinstance(node, dict):
        if "kind" in node:
            yield node

        for value in node.values():
            yield from walk(value)

    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def build_vocab(values, special_token):
    vocab = {special_token: 0}

    for value in sorted(values):
        vocab[value] = len(vocab)

    return vocab

def main():
    kinds = set()
    opcodes = set()
    cast_kinds = set()

    files = list(AST_DIR.glob("*.json"))

    print(f"Scanning {len(files)} AST files...")

    for i, path in enumerate(files, start=1):
        try:
            with path.open("r", encoding="utf-8") as f:
                ast = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {path.name}: {e}")
            continue

        for node in walk(ast):
            kinds.add(node["kind"])

            if "opcode" in node:
                opcodes.add(node["opcode"])

            if "castKind" in node:
                cast_kinds.add(node["castKind"])

        if i % 500 == 0:
            print(f"  processed {i}/{len(files)}")

    vocabularies = {
        "kind": build_vocab(kinds, "<UNK>"),
        "opcode": build_vocab(opcodes, "<NONE>"),
        "castKind": build_vocab(cast_kinds, "<NONE>"),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(vocabularies, f, indent=2)

    print()
    print(f"Saved vocabularies to: {OUTPUT_FILE}")
    print(f"  kind:     {len(vocabularies['kind'])}")
    print(f"  opcode:   {len(vocabularies['opcode'])}")
    print(f"  castKind: {len(vocabularies['castKind'])}")


if __name__ == "__main__":
    main()