from pathlib import Path
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from gnn_ai_code_detector.preprocess import ClangASTConverter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT/"data"/"Code_Dataset"/"HumanVsAi_CodeDataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT/"data"/"c_cpp"/"raw_asts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORCE_REBUILD = False

def process_sample(args):
    df_id, source, language, force_rebuild = args

    output_path = OUTPUT_DIR/f"{df_id}.json"

    if output_path.exists() and not force_rebuild:
        return df_id, "skipped"

    converter = ClangASTConverter()

    suffix = ".c" if language == "C" else ".cpp"

    source_with_headers = converter.append_headers(
        source, language
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(source_with_headers)

        ast = converter.build_ast(temp_path)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(ast, f)

        return df_id, "success"

    except Exception as e:
        return df_id, "failed"

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

def main():
    df = pd.read_csv(DATASET_PATH)

    c_cpp_df = df[
        df["Language"].isin(["C", "C++"])
    ].copy()

    jobs = [
        (df_id, row["Sample_Code"], row["Language"], FORCE_REBUILD)
        for df_id, row in c_cpp_df.iterrows()
    ]

    workers = 12

    print(f"Samples: {len(jobs)}")
    print(f"Workers: {workers}")

    successes = 0
    failures = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_sample, job)
            for job in jobs
        ]

        for i, future in enumerate(as_completed(futures), 1):
            df_id, status = future.result()

            if status == "success":
                successes += 1
            elif status == "failed":
                failures += 1
            else:
                skipped += 1

            print(
                f"[{i}/{len(jobs)}] "
                f"{df_id}: {status}"
            )

    print("\nDone.")
    print(f"Success: {successes}")
    print(f"Failed:  {failures}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    import os
    main()