
import duckdb
import pandas as pd
import numpy as np
import os
import time

from rapidfuzz import fuzz
from xgboost import XGBClassifier

DB_PATH = os.path.abspath("blocking_experiment.duckdb")
MODEL_PATH = os.path.abspath(
    "code/business_entity_resolution/strong_xgboost_model_v2.json"
)
META_PATH = os.path.abspath(
    "code/business_entity_resolution/strong_model_v2_metadata.txt"
)

POSITIVE_TARGET = 300_000
NEGATIVE_TARGET = 900_000

print("=" * 75)
print("CORRECTED XGBOOST V2 TRAINING")
print("=" * 75)

start = time.time()

con = duckdb.connect(DB_PATH)

con.execute("SET memory_limit='6GB'")
con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")

# ------------------------------------------------------------
# 1. Verify candidate pool
# ------------------------------------------------------------

print("\n[1/5] Checking existing candidate pool...")

positive_pool = con.execute("""
SELECT COUNT(*)
FROM train_labeled_v2
WHERE label = 1
""").fetchone()[0]

negative_pool = con.execute("""
SELECT COUNT(*)
FROM train_labeled_v2
WHERE label = 0
""").fetchone()[0]

print(f"Positive pool: {positive_pool:,}")
print(f"Negative pool: {negative_pool:,}")

# ------------------------------------------------------------
# 2. Correct deterministic sampling
# ------------------------------------------------------------

print("\n[2/5] Sampling balanced training data...")

con.execute("DROP TABLE IF EXISTS model_train_v2_corrected")

# Hash filtering gives approximately 1/10 of positives.
# Then LIMIT guarantees exactly the desired target.
con.execute(f"""
CREATE TABLE model_train_v2_corrected AS

SELECT *
FROM (
    SELECT *
    FROM train_labeled_v2
    WHERE label = 1
      AND MOD(
          ABS(
              HASH(
                  source1_entity_id || '|' || matched_entity_id
              )
          ),
          10
      ) = 0
    LIMIT {POSITIVE_TARGET}
)

UNION ALL

SELECT *
FROM (
    SELECT *
    FROM train_labeled_v2
    WHERE label = 0
      AND MOD(
          ABS(
              HASH(
                  source1_entity_id || '|' || matched_entity_id
              )
          ),
          80
      ) = 0
    LIMIT {NEGATIVE_TARGET}
)
""")

train_count = con.execute("""
SELECT COUNT(*)
FROM model_train_v2_corrected
""").fetchone()[0]

pos_count = con.execute("""
SELECT COUNT(*)
FROM model_train_v2_corrected
WHERE label = 1
""").fetchone()[0]

neg_count = con.execute("""
SELECT COUNT(*)
FROM model_train_v2_corrected
WHERE label = 0
""").fetchone()[0]

print(f"Training rows: {train_count:,}")
print(f"Positive:      {pos_count:,}")
print(f"Negative:      {neg_count:,}")

if pos_count < POSITIVE_TARGET:
    raise RuntimeError(
        f"Could not obtain {POSITIVE_TARGET:,} positives. "
        f"Only {pos_count:,} available."
    )

if neg_count < NEGATIVE_TARGET:
    raise RuntimeError(
        f"Could not obtain {NEGATIVE_TARGET:,} negatives. "
        f"Only {neg_count:,} available."
    )

# ------------------------------------------------------------
# 3. Load training text
# ------------------------------------------------------------

print("\n[3/5] Loading training text...")

df = con.execute("""
SELECT
    m.source1_entity_id,
    m.matched_entity_id,
    m.exact_candidate,
    m.shared_name_tokens,
    m.label,

    s1.norm_name AS s1_name,
    s1.norm_address AS s1_address,
    s1.country AS s1_country,

    s2.norm_name AS s2_name,
    s2.norm_address AS s2_address,
    s2.country AS s2_country

FROM model_train_v2_corrected m

INNER JOIN train_s1_v2 s1
    ON m.source1_entity_id = s1.entity_id

INNER JOIN train_s23_v2 s2
    ON m.matched_entity_id = s2.entity_id
""").fetchdf()

print(f"Loaded rows: {len(df):,}")

# ------------------------------------------------------------
# 4. Feature computation
# ------------------------------------------------------------

print("\n[4/5] Computing 25 features...")

def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)

def token_overlap(a, b):
    ta = set(a.split())
    tb = set(b.split())

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / max(len(ta), len(tb))

features = []

for r in df.itertuples(index=False):

    n1 = safe_text(r.s1_name)
    n2 = safe_text(r.s2_name)

    a1 = safe_text(r.s1_address)
    a2 = safe_text(r.s2_address)

    features.append([

        # 1-9: name
        float(n1 == n2 and n1 != ""),
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        fuzz.WRatio(n1, n2) / 100.0,
        token_overlap(n1, n2),
        abs(len(n1) - len(n2)),
        min(len(n1), len(n2)) / max(len(n1), len(n2))
            if max(len(n1), len(n2)) else 0.0,

        # 10-18: address
        float(a1 == a2 and a1 != ""),
        fuzz.ratio(a1, a2) / 100.0,
        fuzz.partial_ratio(a1, a2) / 100.0,
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        fuzz.token_set_ratio(a1, a2) / 100.0,
        fuzz.WRatio(a1, a2) / 100.0,
        token_overlap(a1, a2),
        abs(len(a1) - len(a2)),
        min(len(a1), len(a2)) / max(len(a1), len(a2))
            if max(len(a1), len(a2)) else 0.0,

        # 19: country
        float(
            safe_text(r.s1_country).lower()
            == safe_text(r.s2_country).lower()
        ),

        # 20-23: presence
        float(bool(n1)),
        float(bool(n2)),
        float(bool(a1)),
        float(bool(a2)),

        # 24-25: candidate signals
        float(r.exact_candidate),
        float(r.shared_name_tokens)
    ])

X = np.asarray(features, dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int8)

print("Feature matrix:", X.shape)
print("Positive:", int(y.sum()))
print("Negative:", int((y == 0).sum()))

# ------------------------------------------------------------
# 5. Train corrected V2
# ------------------------------------------------------------

print("\n[5/5] Training corrected XGBoost V2...")

model = XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.04,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.90,
    gamma=0.10,
    reg_alpha=0.10,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=4,
    random_state=42
)

model.fit(X, y)

model.save_model(MODEL_PATH)

with open(META_PATH, "w", encoding="utf-8") as f:
    f.write("CORRECTED FINAL XGBOOST V2 MODEL\n")
    f.write("================================\n")
    f.write(f"Training rows: {train_count}\n")
    f.write(f"Positive rows: {pos_count}\n")
    f.write(f"Negative rows: {neg_count}\n")
    f.write("Candidate strategy: exact + token Top-50\n")
    f.write("Features: 25\n")
    f.write("Training uses all available labeled training data.\n")
    f.write("No train/validation split was used.\n")

print("\nCorrected model saved:")
print(MODEL_PATH)

print("\nMetadata saved:")
print(META_PATH)

print("\nRuntime:", round(time.time() - start, 2), "sec")

con.close()

print("=" * 75)
print("CORRECTED V2 COMPLETE")
print("=" * 75)
