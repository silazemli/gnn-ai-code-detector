from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

def process_ast(
        input_path: Path, output_path: Path,
        force_rebuild: bool, transforms: list[Callable]
    ):
    if output_path.exists() and not force_rebuild:
        return input_path.name, "skipped"

    with input_path.open("r", encoding="utf-8") as f:
        ast = json.load(f)

    for transform in transforms:
        ast = transform(ast)

    temp_path = output_path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(ast, f, indent=2)

    return input_path.name, "processed"

def run_ast_pipeline(
    input_dir: Path,
    output_dir: Path,
    transforms: list[Callable],
    force_rebuild: bool = False,
    workers: int | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.json"))

    print(f"Found {len(files)} ASTs.")

    processed = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_ast, input_path,
                output_dir/input_path.name,
                force_rebuild,
                transforms): input_path
            for input_path in files
        }

        for i, future in enumerate(as_completed(futures), 1):
            input_path = futures[future]

            try:
                _, status = future.result()

                if status == "processed":
                    processed += 1
                else:
                    skipped += 1

                if i % 100 == 0 or i == len(files):
                    print(
                        f"[{i}/{len(files)}] "
                        f"processed={processed}, "
                        f"skipped={skipped}, "
                        f"failed={failed}"
                    )

            except Exception as e:
                failed += 1
                print(
                    f"FAILED: {input_path.name}: {e}"
                )

    print(f"\nProcessed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")