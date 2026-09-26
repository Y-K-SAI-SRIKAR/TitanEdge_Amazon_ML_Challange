"""
FINAL V2 INFERENCE
Corrected candidate-tier insertion and final inference.

Model:
    strong_xgboost_model_v2.json

Features:
    25 features

Output:
    output/matching_results.tsv
"""

import os
import time

import duckdb
import pandas as pd
from rapidfuzz import fuzz
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DB_PATH = os.getenv(
    "DB_PATH",
    "blocking_experiment.duckdb"
)

MODEL_PATH = (
    "code/business_entity_resolution/"
    "strong_xgboost_model_v2.json"
)

OUTPUT_PATH = (
    "output/matching_results.tsv"
)

BATCH_SIZE = 25_000

MEMORY_LIMIT = "6GB"
THREADS = 4

PREDICTION_THRESHOLD = 0.90


# ============================================================
# MODEL FEATURE ORDER
# ============================================================

MODEL_FEATURES = [
    "name_exact",
    "name_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "name_wratio",
    "name_token_overlap",
    "name_length_diff",
    "name_length_ratio",

    "address_exact",
    "address_ratio",
    "address_partial",
    "address_token_sort",
    "address_token_set",
    "address_wratio",
    "address_token_overlap",
    "address_length_diff",
    "address_length_ratio",

    "country_match",

    "s1_name_present",
    "s2_name_present",

    "s1_address_present",
    "s2_address_present",

    "exact_candidate",
    "shared_name_tokens",
]


# ============================================================
# STRING HELPER
# ============================================================

def safe_string(value):

    if pd.isna(value):
        return ""

    return str(value)


# ============================================================
# FUZZY FEATURES
# ============================================================

def fuzzy_features(
    s1_name,
    s23_name,
    s1_address,
    s23_address
):

    s1_name = safe_string(s1_name)
    s23_name = safe_string(s23_name)

    s1_address = safe_string(s1_address)
    s23_address = safe_string(s23_address)


    # ========================================================
    # NAME
    # ========================================================

    if s1_name and s23_name:

        name_exact = int(
            s1_name == s23_name
        )

        name_ratio = fuzz.ratio(
            s1_name,
            s23_name
        )

        name_partial = fuzz.partial_ratio(
            s1_name,
            s23_name
        )

        name_token_sort = fuzz.token_sort_ratio(
            s1_name,
            s23_name
        )

        name_token_set = fuzz.token_set_ratio(
            s1_name,
            s23_name
        )

        name_wratio = fuzz.WRatio(
            s1_name,
            s23_name
        )

    else:

        name_exact = 0
        name_ratio = 0.0
        name_partial = 0.0
        name_token_sort = 0.0
        name_token_set = 0.0
        name_wratio = 0.0


    # ========================================================
    # NAME TOKEN OVERLAP
    # ========================================================

    name_tokens_a = set(
        s1_name.split()
    )

    name_tokens_b = set(
        s23_name.split()
    )

    if name_tokens_a:

        name_token_overlap = (
            len(
                name_tokens_a &
                name_tokens_b
            )
            /
            len(name_tokens_a)
        )

    else:

        name_token_overlap = 0.0


    # ========================================================
    # NAME LENGTH
    # ========================================================

    name_length_diff = abs(
        len(s1_name)
        -
        len(s23_name)
    )

    if s1_name and s23_name:

        name_length_ratio = (
            min(
                len(s1_name),
                len(s23_name)
            )
            /
            max(
                len(s1_name),
                len(s23_name)
            )
        )

    else:

        name_length_ratio = 0.0


    # ========================================================
    # ADDRESS
    # ========================================================

    if s1_address and s23_address:

        address_exact = int(
            s1_address == s23_address
        )

        address_ratio = fuzz.ratio(
            s1_address,
            s23_address
        )

        address_partial = fuzz.partial_ratio(
            s1_address,
            s23_address
        )

        address_token_sort = fuzz.token_sort_ratio(
            s1_address,
            s23_address
        )

        address_token_set = fuzz.token_set_ratio(
            s1_address,
            s23_address
        )

        address_wratio = fuzz.WRatio(
            s1_address,
            s23_address
        )

    else:

        address_exact = 0
        address_ratio = 0.0
        address_partial = 0.0
        address_token_sort = 0.0
        address_token_set = 0.0
        address_wratio = 0.0


    # ========================================================
    # ADDRESS TOKEN OVERLAP
    # ========================================================

    address_tokens_a = set(
        s1_address.split()
    )

    address_tokens_b = set(
        s23_address.split()
    )

    if address_tokens_a:

        address_token_overlap = (
            len(
                address_tokens_a &
                address_tokens_b
            )
            /
            len(address_tokens_a)
        )

    else:

        address_token_overlap = 0.0


    # ========================================================
    # ADDRESS LENGTH
    # ========================================================

    address_length_diff = abs(
        len(s1_address)
        -
        len(s23_address)
    )

    if s1_address and s23_address:

        address_length_ratio = (
            min(
                len(s1_address),
                len(s23_address)
            )
            /
            max(
                len(s1_address),
                len(s23_address)
            )
        )

    else:

        address_length_ratio = 0.0


    return {

        "name_exact":
            name_exact,

        "name_ratio":
            name_ratio,

        "name_partial":
            name_partial,

        "name_token_sort":
            name_token_sort,

        "name_token_set":
            name_token_set,

        "name_wratio":
            name_wratio,

        "name_token_overlap":
            name_token_overlap,

        "name_length_diff":
            name_length_diff,

        "name_length_ratio":
            name_length_ratio,


        "address_exact":
            address_exact,

        "address_ratio":
            address_ratio,

        "address_partial":
            address_partial,

        "address_token_sort":
            address_token_sort,

        "address_token_set":
            address_token_set,

        "address_wratio":
            address_wratio,

        "address_token_overlap":
            address_token_overlap,

        "address_length_diff":
            address_length_diff,

        "address_length_ratio":
            address_length_ratio,
    }


# ============================================================
# START
# ============================================================

overall_start = time.time()

print("=" * 80)
print("FINAL V2 INFERENCE")
print("=" * 80)

print(
    f"Database : {DB_PATH}"
)

print(
    f"Model    : {MODEL_PATH}"
)

print(
    f"Batch    : {BATCH_SIZE:,}"
)

print(
    f"Threshold: {PREDICTION_THRESHOLD}"
)


# ============================================================
# CONNECT
# ============================================================

con = duckdb.connect(
    DB_PATH
)

con.execute(
    f"SET memory_limit='{MEMORY_LIMIT}'"
)

con.execute(
    f"SET threads={THREADS}"
)

con.execute(
    "SET preserve_insertion_order=false"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 XGBoost model...")

model = XGBClassifier()

model.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# CHECK TABLE
# ============================================================

total_candidates = con.execute(
    """
    SELECT COUNT(*)
    FROM final_inference_features
    """
).fetchone()[0]


print(
    f"\nTotal test candidates: "
    f"{total_candidates:,}"
)


# ============================================================
# CANDIDATE TIERS
# ============================================================

print("\n")
print("=" * 80)
print("TEST CANDIDATE TIERS")
print("=" * 80)

tier = con.execute(
    """
    SELECT

        SUM(
            CASE
                WHEN exact_candidate = 1
                THEN 1
                ELSE 0
            END
        ),

        SUM(
            CASE
                WHEN exact_candidate = 0
                 AND shared_name_tokens = 0
                THEN 1
                ELSE 0
            END
        ),

        SUM(
            CASE
                WHEN exact_candidate = 0
                 AND shared_name_tokens = 1
                THEN 1
                ELSE 0
            END
        ),

        SUM(
            CASE
                WHEN exact_candidate = 0
                 AND shared_name_tokens >= 2
                THEN 1
                ELSE 0
            END
        )

    FROM final_inference_features
    """
).fetchone()

exact_count = tier[0] or 0
zero_count = tier[1] or 0
one_count = tier[2] or 0
multi_count = tier[3] or 0

print(
    f"Exact candidates    : {exact_count:,}"
)

print(
    f"0-token candidates  : {zero_count:,}"
)

print(
    f"1-token candidates  : {one_count:,}"
)

print(
    f"2+ token candidates : {multi_count:,}"
)


# ============================================================
# CREATE FINAL ML TABLE
# ============================================================

print("\n")
print("=" * 80)
print("BUILDING FINAL ML CANDIDATE POOL")
print("=" * 80)

con.execute(
    """
    DROP TABLE IF EXISTS final_ml_candidates
    """
)


# ------------------------------------------------------------
# IMPORTANT:
# Explicitly select columns.
# This prevents the 15-vs-13 column INSERT error.
# ------------------------------------------------------------

con.execute(
    """
    CREATE TABLE final_ml_candidates AS

    SELECT

        source1_entity_id,
        matched_entity_id,

        s1_name,
        s23_name,

        s1_address,
        s23_address,

        country_match,

        both_names_present,
        both_addresses_present,

        name_length_diff,
        address_length_diff,

        exact_candidate,
        shared_name_tokens,

        CAST(
            NULL AS INTEGER
        ) AS reserved_col_1,

        CAST(
            NULL AS INTEGER
        ) AS reserved_col_2

    FROM final_inference_features

    WHERE
        exact_candidate = 1

        OR shared_name_tokens = 0

        OR shared_name_tokens >= 2
    """
)


mandatory_count = con.execute(
    """
    SELECT COUNT(*)
    FROM final_ml_candidates
    """
).fetchone()[0]


print(
    f"Mandatory candidates: "
    f"{mandatory_count:,}"
)


# ============================================================
# 1-TOKEN NUMBERING
# ============================================================

print("\n")
print("=" * 80)
print("PROCESSING 1-TOKEN CANDIDATES")
print("=" * 80)

print(
    "Training-derived pre-filter:"
)

print(
    "    address_ratio >= 65"
)

print(
    "    OR name_ratio >= 70"
)


con.execute(
    """
    DROP TABLE IF EXISTS one_token_numbered
    """
)


print(
    "\nCreating numbered 1-token candidate table..."
)


con.execute(
    """
    CREATE TABLE one_token_numbered AS

    SELECT

        ROW_NUMBER() OVER (
            ORDER BY
                source1_entity_id,
                matched_entity_id
        ) AS rn,

        source1_entity_id,
        matched_entity_id,

        s1_name,
        s23_name,

        s1_address,
        s23_address,

        country_match,

        both_names_present,
        both_addresses_present,

        name_length_diff,
        address_length_diff,

        exact_candidate,
        shared_name_tokens

    FROM final_inference_features

    WHERE
        exact_candidate = 0
        AND shared_name_tokens = 1
    """
)


one_token_total = con.execute(
    """
    SELECT COUNT(*)
    FROM one_token_numbered
    """
).fetchone()[0]


print(
    f"1-token candidates: "
    f"{one_token_total:,}"
)


# ============================================================
# PROCESS 1-TOKEN CANDIDATES
# ============================================================

processed = 0
kept = 0

filter_start = time.time()


while processed < one_token_total:

    start_rn = processed + 1

    end_rn = min(
        processed + BATCH_SIZE,
        one_token_total
    )


    batch = con.execute(
        f"""
        SELECT *

        FROM one_token_numbered

        WHERE
            rn >= {start_rn}
            AND rn <= {end_rn}

        ORDER BY rn
        """
    ).fetchdf()


    if batch.empty:
        break


    keep_rows = []


    for _, row in batch.iterrows():

        s1_name = safe_string(
            row["s1_name"]
        )

        s23_name = safe_string(
            row["s23_name"]
        )

        s1_address = safe_string(
            row["s1_address"]
        )

        s23_address = safe_string(
            row["s23_address"]
        )


        if s1_name and s23_name:

            name_ratio = fuzz.ratio(
                s1_name,
                s23_name
            )

        else:

            name_ratio = 0


        if s1_address and s23_address:

            address_ratio = fuzz.ratio(
                s1_address,
                s23_address
            )

        else:

            address_ratio = 0


        if (
            address_ratio >= 65
            or
            name_ratio >= 70
        ):

            keep_rows.append(
                (
                    row["source1_entity_id"],
                    row["matched_entity_id"],
                    s1_name,
                    s23_name,
                    s1_address,
                    s23_address,
                    int(row["country_match"]),
                    int(row["both_names_present"]),
                    int(row["both_addresses_present"]),
                    int(row["name_length_diff"]),
                    int(row["address_length_diff"]),
                    int(row["exact_candidate"]),
                    int(row["shared_name_tokens"]),
                    0,
                    0
                )
            )


    if keep_rows:

        kept_df = pd.DataFrame(
            keep_rows,
            columns=[
                "source1_entity_id",
                "matched_entity_id",

                "s1_name",
                "s23_name",

                "s1_address",
                "s23_address",

                "country_match",

                "both_names_present",
                "both_addresses_present",

                "name_length_diff",
                "address_length_diff",

                "exact_candidate",
                "shared_name_tokens",

                "reserved_col_1",
                "reserved_col_2",
            ]
        )


        con.register(
            "kept_batch",
            kept_df
        )


        # ----------------------------------------------------
        # EXPLICIT INSERT COLUMN LIST
        # ----------------------------------------------------

        con.execute(
            """
            INSERT INTO final_ml_candidates (

                source1_entity_id,
                matched_entity_id,

                s1_name,
                s23_name,

                s1_address,
                s23_address,

                country_match,

                both_names_present,
                both_addresses_present,

                name_length_diff,
                address_length_diff,

                exact_candidate,
                shared_name_tokens,

                reserved_col_1,
                reserved_col_2

            )

            SELECT

                source1_entity_id,
                matched_entity_id,

                s1_name,
                s23_name,

                s1_address,
                s23_address,

                country_match,

                both_names_present,
                both_addresses_present,

                name_length_diff,
                address_length_diff,

                exact_candidate,
                shared_name_tokens,

                reserved_col_1,
                reserved_col_2

            FROM kept_batch
            """
        )


        con.unregister(
            "kept_batch"
        )


        kept += len(
            kept_df
        )


    processed = end_rn


    if (
        processed == one_token_total
        or
        processed % (
            BATCH_SIZE * 10
        ) == 0
    ):

        elapsed = (
            time.time()
            -
            filter_start
        )

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            one_token_total -
            processed
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"Processed "
            f"{processed:,}/"
            f"{one_token_total:,}"
            f" ({processed / one_token_total:.1%})"
            f" | kept {kept:,}"
            f" | ETA {eta / 60:.1f} min"
        )


# ============================================================
# FINAL ML COUNT
# ============================================================

ml_count = con.execute(
    """
    SELECT COUNT(*)
    FROM final_ml_candidates
    """
).fetchone()[0]


print("\n")
print("=" * 80)
print("FINAL ML POOL READY")
print("=" * 80)

print(
    f"Original candidates : "
    f"{total_candidates:,}"
)

print(
    f"ML candidates       : "
    f"{ml_count:,}"
)

print(
    f"1-token retained    : "
    f"{kept:,}"
)

print(
    f"Reduction           : "
    f"{100 * (1 - ml_count / total_candidates):.2f}%"
)


# ============================================================
# NUMBER ML CANDIDATES
# ============================================================

print(
    "\nNumbering final ML candidates..."
)

con.execute(
    """
    DROP TABLE IF EXISTS final_ml_numbered
    """
)


con.execute(
    """
    CREATE TABLE final_ml_numbered AS

    SELECT

        ROW_NUMBER() OVER (
            ORDER BY
                source1_entity_id,
                matched_entity_id
        ) AS rn,

        *

    FROM final_ml_candidates
    """
)


# ============================================================
# SCORE TABLE
# ============================================================

con.execute(
    """
    DROP TABLE IF EXISTS final_scored_candidates
    """
)


con.execute(
    """
    CREATE TABLE final_scored_candidates (

        source1_entity_id VARCHAR,
        matched_entity_id VARCHAR,
        probability DOUBLE

    )
    """
)


# ============================================================
# XGBOOST INFERENCE
# ============================================================

print("\n")
print("=" * 80)
print("RUNNING V2 XGBOOST INFERENCE")
print("=" * 80)

print(
    f"Candidates : {ml_count:,}"
)

print(
    f"Threshold  : {PREDICTION_THRESHOLD}"
)


inference_start = time.time()

processed = 0
positive_count = 0


while processed < ml_count:

    start_rn = processed + 1

    end_rn = min(
        processed + BATCH_SIZE,
        ml_count
    )


    batch = con.execute(
        f"""
        SELECT

            source1_entity_id,
            matched_entity_id,

            s1_name,
            s23_name,

            s1_address,
            s23_address,

            country_match,

            exact_candidate,
            shared_name_tokens

        FROM final_ml_numbered

        WHERE
            rn >= {start_rn}
            AND rn <= {end_rn}

        ORDER BY rn
        """
    ).fetchdf()


    if batch.empty:
        break


    feature_rows = []


    for _, row in batch.iterrows():

        fuzzy = fuzzy_features(

            row["s1_name"],
            row["s23_name"],

            row["s1_address"],
            row["s23_address"]
        )


        s1_name = safe_string(
            row["s1_name"]
        )

        s23_name = safe_string(
            row["s23_name"]
        )

        s1_address = safe_string(
            row["s1_address"]
        )

        s23_address = safe_string(
            row["s23_address"]
        )


        features = {

            **fuzzy,

            "country_match":
                int(row["country_match"]),

            "s1_name_present":
                int(bool(s1_name)),

            "s2_name_present":
                int(bool(s23_name)),

            "s1_address_present":
                int(bool(s1_address)),

            "s2_address_present":
                int(bool(s23_address)),

            "exact_candidate":
                int(row["exact_candidate"]),

            "shared_name_tokens":
                int(row["shared_name_tokens"]),
        }


        feature_rows.append(
            features
        )


    X = pd.DataFrame(
        feature_rows,
        columns=MODEL_FEATURES
    )


    probabilities = model.predict_proba(
        X
    )[:, 1]


    mask = (
        probabilities >=
        PREDICTION_THRESHOLD
    )


    if mask.any():

        matched = pd.DataFrame(
            {
                "source1_entity_id":
                    batch.loc[
                        mask,
                        "source1_entity_id"
                    ].values,

                "matched_entity_id":
                    batch.loc[
                        mask,
                        "matched_entity_id"
                    ].values,

                "probability":
                    probabilities[mask]
            }
        )


        con.register(
            "matched_batch",
            matched
        )


        con.execute(
            """
            INSERT INTO final_scored_candidates

            SELECT
                source1_entity_id,
                matched_entity_id,
                probability

            FROM matched_batch
            """
        )


        con.unregister(
            "matched_batch"
        )


        positive_count += len(
            matched
        )


    processed = end_rn


    if (
        processed == ml_count
        or
        processed % (
            BATCH_SIZE * 10
        ) == 0
    ):

        elapsed = (
            time.time()
            -
            inference_start
        )

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            ml_count -
            processed
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"Processed "
            f"{processed:,}/"
            f"{ml_count:,}"
            f" ({processed / ml_count:.1%})"
            f" | matches {positive_count:,}"
            f" | ETA {eta / 60:.1f} min"
        )


# ============================================================
# CREATE OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("CREATING FINAL OUTPUT")
print("=" * 80)


predicted = con.execute(
    """
    SELECT

        source1_entity_id,

        STRING_AGG(
            matched_entity_id,
            ','
            ORDER BY matched_entity_id
        ) AS matched_entity_ids

    FROM final_scored_candidates

    GROUP BY source1_entity_id
    """
).fetchdf()


all_s1 = con.execute(
    """
    SELECT
        entity_id AS source1_entity_id

    FROM test_source1

    ORDER BY entity_id
    """
).fetchdf()


results = all_s1.merge(
    predicted,
    on="source1_entity_id",
    how="left"
)


results[
    "matched_entity_ids"
] = results[
    "matched_entity_ids"
].fillna("")


# ============================================================
# VALIDATION
# ============================================================

if len(results) != len(all_s1):

    raise RuntimeError(
        "Incorrect number of output rows."
    )


if not results[
    "source1_entity_id"
].is_unique:

    raise RuntimeError(
        "Duplicate Source 1 IDs detected."
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "output",
    exist_ok=True
)


results.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

matched_entities = (
    results[
        "matched_entity_ids"
    ] != ""
).sum()


empty_entities = (
    results[
        "matched_entity_ids"
    ] == ""
).sum()


total_pairs = 0


for value in results[
    "matched_entity_ids"
]:

    if value:

        total_pairs += len(
            value.split(",")
        )


elapsed = (
    time.time()
    -
    overall_start
)


print("\n")
print("=" * 80)
print("FINAL INFERENCE COMPLETE")
print("=" * 80)

print(
    f"Output file         : "
    f"{OUTPUT_PATH}"
)

print(
    f"Source 1 rows       : "
    f"{len(results):,}"
)

print(
    f"Matched entities    : "
    f"{matched_entities:,}"
)

print(
    f"Empty entities      : "
    f"{empty_entities:,}"
)

print(
    f"Predicted pairs     : "
    f"{total_pairs:,}"
)

print(
    f"Runtime             : "
    f"{elapsed / 60:.2f} minutes"
)

print("\n")
print(
    "Run the official validator:"
)

print(
    r"python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test"
)

print("=" * 80)


con.close()