from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT/"data"/"c_cpp"/"cut_asts"
OUTPUT_DIR = PROJECT_ROOT/"data"/"c_cpp"/"clean_asts"

FORCE_REBUILD = True

from gnn_ai_code_detector.preprocess import ClangASTConverter
converter = ClangASTConverter("") # Clang is never invoked so whatever

def process_file(raw_path: Path):
    output_path = OUTPUT_DIR/raw_path.name

    if output_path.exists() and not FORCE_REBUILD:
        return "skipped"

    with raw_path.open("r", encoding="utf-8") as f:
        ast = json.load(f)

    clean_ast = \
        converter.remove_metadata(
        converter.remove_irrelevant_nodes(ast)
    )

    temp_path = output_path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(clean_ast, f, indent=2)

    temp_path.replace(output_path)

    return "processed"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DIR.glob("*.json"))

    print(f"Found {len(files)} ASTs.")

    processed = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, path): path
            for path in files
        }

        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]

            try:
                status = future.result()

                if status == "processed":
                    processed += 1
                else:
                    skipped += 1

                if i % 100 == 0 or i == len(files):
                    print(
                        f"[{i}/{len(files)}] "
                        f"processed={processed}, "
                        f"skipped={skipped}"
                    )

            except Exception as e:
                failed += 1
                print(f"FAILED: {path.name}: {e}")

    print(f"\nProcessed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

if __name__ == "__main__":
    main()