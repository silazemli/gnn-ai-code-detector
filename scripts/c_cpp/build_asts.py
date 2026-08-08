from pathlib import Path
import json
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from gnn_ai_code_detector.preprocess import CCppPreprocessor

def process_sample(
        df_id: int, source: str, 
        language: str, output_dir: Path,
        force_rebuild: bool, preprocessor: CCppPreprocessor):
    
    output_path = output_dir/f"{df_id}.json"

    if output_path.exists() and not force_rebuild:
        return df_id, "skipped"

    suffix = ".c" if language == "C" else ".cpp"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            encoding="utf-8",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(
                preprocessor.append_headers(
                    source, language
                )
            )

        ast = preprocessor.build_ast(temp_path)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(ast, f)

        return df_id, "success"

    except Exception as e:
        return df_id, "failed"

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

def build_asts(
    csv_path: Path, output_dir: Path,
    preprocessor: CCppPreprocessor,
    force_rebuild: bool = False,
    workers: int = 8
    ):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    c_cpp_df = df[df["Language"].isin(["C", "C++"])]

    print(f"Found {len(c_cpp_df)} samples.")

    successes = 0
    failures = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_sample,
                df_id, row["Sample_Code"],
                row["Language"], output_dir,
                force_rebuild, preprocessor
            ): df_id
            for df_id, row in c_cpp_df.iterrows()
        }

        for i, future in enumerate(as_completed(futures), 1):
            df_id = futures[future]

            try:
                _, status = future.result()

                if status == "success":
                    successes += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failures += 1

            except Exception as e:
                failures += 1
                print(f"FAILED: {df_id}: {e}")

            if i % 100 == 0 or i == len(c_cpp_df):
                print(
                    f"[{i}/{len(c_cpp_df)}] "
                    f"success={successes}, "
                    f"skipped={skipped}, "
                    f"failed={failures}"
                )


    print("\nDone.")
    print(f"Success: {successes}")
    print(f"Failed:  {failures}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    CSV_PATH = PROJECT_ROOT/"data"/"Code_Dataset"/"HumanVsAi_CodeDataset.csv"
    OUTPUT_DIR = PROJECT_ROOT/"data"/"c_cpp"/"raw_asts"

    CLANG_PATH = "C:/Program Files/LLVM/bin/clang.exe"

    FORCE_REBUILD = False

    build_asts(
        CSV_PATH, OUTPUT_DIR,
        CCppPreprocessor(CLANG_PATH),
        FORCE_REBUILD
    )