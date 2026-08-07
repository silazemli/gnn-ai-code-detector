from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def get_huvsai_split(
        csv_path: Path,
        language: str,
        random_state: int = 42,
    ):
    df = pd.read_csv(csv_path)

    if language == "C/C++":
        langs = ["C", "C++"]
    elif language == "Python":
        langs = ["Python"]
    else:
        raise ValueError("Unrecognised language")

    df = df[df["Language"].isin(langs)].copy()

    problem_df = df[["problem_id"]].drop_duplicates()

    # no stratification because the original dataset is balanced
    train_problems, test_problems = train_test_split(
        problem_df, test_size=0.2,
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

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATASET_PATH = (
        PROJECT_ROOT/"data"/"Code_Dataset"/"HumanVsAi_CodeDataset.csv"
    )

    train, test = get_huvsai_split(DATASET_PATH, "C/C++")
    df = pd.read_csv(DATASET_PATH)

    train_df = df.loc[train]
    test_df = df.loc[test]

    print("TRAIN")
    print(train_df["Generated"].value_counts())
    print(train_df["Generated"].value_counts(normalize=True))

    print("\nTEST")
    print(test_df["Generated"].value_counts())
    print(test_df["Generated"].value_counts(normalize=True))
