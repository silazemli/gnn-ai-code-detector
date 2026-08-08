from pathlib import Path

from .ast_pipeline import run_ast_pipeline
from gnn_ai_code_detector.preprocess import CCppPreprocessor

def clean_asts(
    input_dir: Path,
    output_dir: Path,
    preprocessor: CCppPreprocessor,
    force_rebuild: bool = False,
    workers: int = 8,
):
    run_ast_pipeline(
        input_dir, output_dir, force_rebuild,
        [preprocessor.remove_irrelevant_nodes, preprocessor.remove_metadata],
        workers
    )


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    INPUT_DIR = PROJECT_ROOT/"data"/"c_cpp"/"cut_asts"
    OUTPUT_DIR = PROJECT_ROOT/"data"/"c_cpp"/"clean_asts"

    FORCE_REBUILD = False

    clean_asts(
        INPUT_DIR, OUTPUT_DIR,
        CCppPreprocessor(),
        FORCE_REBUILD,
    )