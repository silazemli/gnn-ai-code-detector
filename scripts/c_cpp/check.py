from pathlib import Path
import json

from gnn_ai_code_detector.preprocess import ClangASTConverter

AST_PATH = Path(__file__).parent / "7.json"

with AST_PATH.open("r", encoding="utf-8") as f:
    ast = json.load(f)

converter = ClangASTConverter("")
graph = converter.construct_graph(ast)
graph = converter.handle_external_references(graph)

print(f"Nodes: {len(graph)}")

child_edges = sum(len(edges["edges"]["children"]) for edges in graph.values())
reference_edges = sum(len(edges["edges"]["references"]) for edges in graph.values())

print(f"Child edges: {child_edges}")
print(f"Reference edges: {reference_edges}")
print(f"Total edges: {child_edges + reference_edges}")

for node_id, node in graph.items():
    print(f"\n{node_id}")
    print(f"  features:   {node['features']}")
    print(f"  children:   {node['edges']['children']}")
    print(f"  references: {node['edges']['references']}")

node_ids = set(graph)

missing_references = [
    (node_id, ref)
    for node_id, data in graph.items()
    for ref in data["edges"]["references"]
    if ref not in node_ids
]

print(f"Missing reference targets: {len(missing_references)}")

missing_references = [
    (node_id, ref)
    for node_id, data in graph.items()
    for ref in data["edges"]["references"]
    if ref not in graph
]

print(f"Missing reference targets: {len(missing_references)}")

for source, target in missing_references:
    print(f"  {source} -> {target}")