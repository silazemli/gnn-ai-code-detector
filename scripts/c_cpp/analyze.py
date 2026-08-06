from pathlib import Path
from collections import Counter, defaultdict
import json
import random

OUTPUT_FILE = Path(__file__).parent/"analysis.txt"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AST_DIR = PROJECT_ROOT/"data"/"c_cpp"/"clean_asts"

files = list(AST_DIR.glob("*.json"))

NUM_FILES = len(files)

if len(files) > NUM_FILES:
    files = random.sample(files, NUM_FILES)

node_kinds = Counter()
field_counts = Counter()
field_values = defaultdict(Counter)

files_processed = 0
nodes_processed = 0

def walk(node):
    if isinstance(node, dict):
        if "kind" in node:
            yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)

for path in files:
    try:
        with path.open("r", encoding="utf-8") as f:
            ast = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Skipping {path.name}: {e}")
        continue

    files_processed += 1
    for node in walk(ast):
        nodes_processed += 1
        node_kinds[node["kind"]] += 1
        for key, value in node.items():
            field_counts[key] += 1
            if isinstance(value, (str, int, float, bool)) or value is None:
                field_values[key][str(value)] += 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    print("=" * 80, file=out)
    print(f"Files processed: {files_processed}", file=out)
    print(f"Nodes inspected: {nodes_processed}", file=out)
    print("=" * 80, file=out)
    print("\nNODE KINDS", file=out)
    print("-" * 80, file=out)
    for kind, count in node_kinds.most_common():
        print(f"{count:8}  {kind}", file=out)

    print("\nFIELDS", file=out)
    print("-" * 80, file=out)
    for field, count in field_counts.most_common():
        percentage = 100 * count / nodes_processed
        print(f"{count:8}  {percentage:6.2f}%  {field}", file=out)

    print("\nSCALAR FIELD VALUES", file=out)
    print("-" * 80, file=out)
    for field, values in sorted(field_values.items()):
        print(f"\n[{field}]", file=out)
        for value, count in values.most_common(30):
            print(f"  {count:8}  {value}", file=out)