import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# This file is:
# student_resource/code/business_entity_resolution/src/
#
# Dataset is:
# student_resource/dataset/

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    train_s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t"
    )

    train_s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t"
    )

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )

    test_s1 = pd.read_csv(
        TEST_DIR / "test_source1.tsv",
        sep="\t"
    )

    test_s2 = pd.read_csv(
        TEST_DIR / "test_source2.tsv",
        sep="\t"
    )

    test_s3 = pd.read_csv(
        TEST_DIR / "test_source3.tsv",
        sep="\t"
    )

    return (
        train_s1,
        train_s2,
        train_s3,
        ground_truth,
        test_s1,
        test_s2,
        test_s3
    )


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def inspect_dataframe(name, df):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nMissing values:")
    missing = df.isnull().sum()

    for col, count in missing.items():
        percentage = (count / len(df)) * 100
        print(
            f"  {col:<20} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    print("\nSample:")
    print(df.head(3).to_string(index=False))


# ============================================================
# GROUND TRUTH ANALYSIS
# ============================================================

def analyze_ground_truth(gt):

    print("\n" + "=" * 70)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 70)

    print(f"Ground truth rows: {len(gt):,}")

    # Convert missing values to empty strings
    matches = gt["matched_entity_ids"].fillna("").astype(str)

    # Number of matched entities for each Source 1 entity
    match_counts = matches.apply(
        lambda x: 0 if not x.strip() else len(x.split(","))
    )

    print("\nMatch count distribution:")
    print(match_counts.value_counts().sort_index())

    singleton_count = (match_counts == 0).sum()
    matched_count = (match_counts > 0).sum()

    print("\nSource 1 entities:")
    print(f"  Total entities       : {len(gt):,}")
    print(f"  Entities with matches: {matched_count:,}")
    print(f"  Singletons            : {singleton_count:,}")

    if len(gt) > 0:
        print(
            f"  Singleton percentage : "
            f"{singleton_count / len(gt) * 100:.2f}%"
        )

    print("\nMaximum matches for one Source 1 entity:")
    print(f"  {match_counts.max()}")

    print("\nAverage matches per Source 1 entity:")
    print(f"  {match_counts.mean():.3f}")


# ============================================================
# COUNTRY ANALYSIS
# ============================================================

def analyze_countries(name, df):

    print("\n" + "=" * 70)
    print(f"COUNTRY DISTRIBUTION — {name}")
    print("=" * 70)

    print(df["country"].value_counts(dropna=False).to_string())


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(name, df):

    print("\n" + "=" * 70)
    print(f"DUPLICATE ANALYSIS — {name}")
    print("=" * 70)

    for col in [
        "entity_id",
        "business_name",
        "business_address"
    ]:

        if col not in df.columns:
            continue

        duplicate_count = df[col].duplicated().sum()

        print(
            f"{col:<20}: "
            f"{duplicate_count:,} duplicate rows"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(" AMAZON ML — BUSINESS ENTITY RESOLUTION")
    print(" DATASET AUDIT")
    print("=" * 70)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Train path  : {TRAIN_DIR}")
    print(f"Test path   : {TEST_DIR}")

    (
        train_s1,
        train_s2,
        train_s3,
        ground_truth,
        test_s1,
        test_s2,
        test_s3
    ) = load_data()

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    inspect_dataframe("TRAIN SOURCE 1", train_s1)
    inspect_dataframe("TRAIN SOURCE 2", train_s2)
    inspect_dataframe("TRAIN SOURCE 3", train_s3)
    inspect_dataframe("GROUND TRUTH", ground_truth)

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    inspect_dataframe("TEST SOURCE 1", test_s1)
    inspect_dataframe("TEST SOURCE 2", test_s2)
    inspect_dataframe("TEST SOURCE 3", test_s3)

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    analyze_ground_truth(ground_truth)

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    analyze_countries("TRAIN SOURCE 1", train_s1)
    analyze_countries("TRAIN SOURCE 2", train_s2)
    analyze_countries("TRAIN SOURCE 3", train_s3)

    analyze_countries("TEST SOURCE 1", test_s1)
    analyze_countries("TEST SOURCE 2", test_s2)
    analyze_countries("TEST SOURCE 3", test_s3)

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    analyze_duplicates("TRAIN SOURCE 1", train_s1)
    analyze_duplicates("TRAIN SOURCE 2", train_s2)
    analyze_duplicates("TRAIN SOURCE 3", train_s3)

    print("\n" + "=" * 70)
    print("DATA AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()