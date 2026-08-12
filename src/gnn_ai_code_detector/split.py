from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def get_huvsai_split(
        df: pd.DataFrame,
        ast_dir: Path,
        language: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
    if language == "CC++":
        langs = ["C", "C++"]
    elif language == "C":
        langs = ["C"]
    elif language == "C++":
        langs = ["C++"]
    elif language == "Python":
        langs = ["Python"]
    else:
        raise ValueError("Unrecognised language")

    df = df[df["Language"].isin(langs)].copy()

    df = df[df.index.map(lambda index: (ast_dir/f"{index}.json").exists())].copy()

    problem_df = df[["problem_id"]].drop_duplicates()

    train_problems, test_problems = train_test_split(
        problem_df, test_size=test_size,
        random_state=random_state
    )

    train_problem_ids = set(train_problems["problem_id"])
    test_problem_ids = set(test_problems["problem_id"])

    train_indices = df[
        df["problem_id"].isin(train_problem_ids)
    ].index.tolist()

    test_indices = df[
        df["problem_id"].isin(test_problem_ids)
    ].index.tolist()

    return train_indices, test_indices