from pathlib import Path

from gnn_ai_code_detector.dataset import CCppDataset
from gnn_ai_code_detector.preprocess import CCppPreprocessor
from gnn_ai_code_detector.split import get_huvsai_split

from scripts.c_cpp.build_asts import build_asts
from scripts.c_cpp.cut_asts import cut_asts
from scripts.c_cpp.clean_asts import clean_asts

BUILD_ASTS = False
CUT_ASTS = False
CLEAN_ASTS = False

LANGUAGE = "C/C++"

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT/"data"

CSV_PATH = DATA_PATH/"Code_Dataset"/"HumanVsAi_CodeDataset.csv"

RAW_AST_DIR = DATA_PATH/"c_cpp"/"raw_asts"
CUT_AST_DIR = DATA_PATH/"c_cpp"/"cut_asts"
CLEAN_AST_DIR = DATA_PATH/"c_cpp"/"clean_asts"

CLANG_PATH = "C:/Program Files/LLVM/bin/clang.exe"
preprocessor = CCppPreprocessor(CLANG_PATH)

FORCE_BUILD = False
FORCE_CUT = False
FORCE_CLEAN = False

WORKERS = 12

if BUILD_ASTS: build_asts(CSV_PATH, RAW_AST_DIR, preprocessor, FORCE_BUILD, WORKERS)
if CUT_ASTS: cut_asts(RAW_AST_DIR, CUT_AST_DIR, preprocessor, FORCE_CUT, WORKERS)
if CLEAN_ASTS: clean_asts(CUT_AST_DIR, CLEAN_AST_DIR, preprocessor, FORCE_CLEAN, WORKERS)

train_indices, test_indices = get_huvsai_split(CSV_PATH, LANGUAGE, random_state=RANDOM_STATE)

print(f"Train samples: {len(train_indices)}")
print(f"Test samples:  {len(test_indices)}")

train_dataset = CCppDataset(
    indices=train_indices,
    ast_dir=CLEAN_AST_DIR,
    preprocessor=preprocessor
)

test_dataset = CCppDataset(
    indices=test_indices,
    ast_dir=CLEAN_AST_DIR,
    preprocessor=preprocessor
)

vocab = preprocessor.build_vocabularies(train_dataset)

train_dataset.vocab = vocab
test_dataset.vocab = vocab

