import duckdb
from pathlib import Path
import time


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

DB_PATH = PROJECT_ROOT / "blocking_experiment.duckdb"


# ============================================================
# NORMALIZATION
# ============================================================

# Keep this normalization intentionally conservative.
#
# We want to test whether cheap normalization + exact matching
# is already capable of recovering a large percentage of the
# true matches.
#
# More aggressive normalization/fuzzy matching comes later.

NORMALIZE_SQL = """
lower(
    regexp_replace(
        regexp_replace(
            trim(coalesce({column}, '')),
            '[^[:alnum:]]+',
            ' ',
            'g'
        ),
        '\\s+',
        ' ',
        'g'
    )
)
"""


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(" AMAZON ML — BLOCKING EXPERIMENT")
    print("=" * 75)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Train path   : {TRAIN_DIR}")
    print(f"DuckDB file  : {DB_PATH}")

    start_total = time.time()

    con = duckdb.connect(str(DB_PATH))

    # --------------------------------------------------------
    # PERFORMANCE SETTINGS
    # --------------------------------------------------------

    con.execute("""
        SET threads = 8
    """)

    # Allow DuckDB to spill to disk instead of crashing if the
    # working set becomes larger than available RAM.
    con.execute("""
        SET memory_limit = '8GB'
    """)

    # --------------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------------

    print("\n[1/6] Preparing Source 2...")

    start = time.time()

    con.execute(f"""
        CREATE OR REPLACE TABLE source2 AS

        SELECT
            entity_id,

            business_name,

            business_address,

            country,

            {NORMALIZE_SQL.format(column="business_name")}
                AS name_norm,

            {NORMALIZE_SQL.format(column="business_address")}
                AS address_norm

        FROM read_csv(
            '{(TRAIN_DIR / "train_source2.tsv").as_posix()}',
            delim='\\t',
            header=true,
            nullstr=''
        )
    """)

    count_s2 = con.execute(
        "SELECT COUNT(*) FROM source2"
    ).fetchone()[0]

    print(f"Source 2 rows: {count_s2:,}")
    print(f"Time: {time.time() - start:.2f}s")

    # --------------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------------

    print("\n[2/6] Preparing Source 3...")

    start = time.time()

    con.execute(f"""
        CREATE OR REPLACE TABLE source3 AS

        SELECT
            entity_id,

            business_name,

            business_address,

            country,

            {NORMALIZE_SQL.format(column="business_name")}
                AS name_norm,

            {NORMALIZE_SQL.format(column="business_address")}
                AS address_norm

        FROM read_csv(
            '{(TRAIN_DIR / "train_source3.tsv").as_posix()}',
            delim='\\t',
            header=true,
            nullstr=''
        )
    """)

    count_s3 = con.execute(
        "SELECT COUNT(*) FROM source3"
    ).fetchone()[0]

    print(f"Source 3 rows: {count_s3:,}")
    print(f"Time: {time.time() - start:.2f}s")

    # --------------------------------------------------------
    # SOURCE 1
    # --------------------------------------------------------

    print("\n[3/6] Preparing Source 1...")

    start = time.time()

    con.execute(f"""
        CREATE OR REPLACE TABLE source1 AS

        SELECT
            entity_id,

            business_name,

            business_address,

            country,

            {NORMALIZE_SQL.format(column="business_name")}
                AS name_norm,

            {NORMALIZE_SQL.format(column="business_address")}
                AS address_norm

        FROM read_csv(
            '{(TRAIN_DIR / "train_source1.tsv").as_posix()}',
            delim='\\t',
            header=true,
            nullstr=''
        )
    """)

    count_s1 = con.execute(
        "SELECT COUNT(*) FROM source1"
    ).fetchone()[0]

    print(f"Source 1 rows: {count_s1:,}")
    print(f"Time: {time.time() - start:.2f}s")

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    print("\n[4/6] Preparing ground truth...")

    start = time.time()

    con.execute(f"""
        CREATE OR REPLACE TABLE ground_truth_raw AS

        SELECT
            source1_entity_id,
            matched_entity_ids

        FROM read_csv(
            '{(TRAIN_DIR / "train_ground_truth.tsv").as_posix()}',
            delim='\\t',
            header=true,
            nullstr=''
        )
    """)

    # Convert comma-separated match lists into individual rows.
    #
    # Example:
    #
    # S1-001 -> S2-010,S2-020,S3-030
    #
    # becomes:
    #
    # S1-001 -> S2-010
    # S1-001 -> S2-020
    # S1-001 -> S3-030

    con.execute("""
        CREATE OR REPLACE TABLE truth_pairs AS

        SELECT
            source1_entity_id,
            trim(matched_id) AS matched_entity_id

        FROM ground_truth_raw,
        UNNEST(
            CASE
                WHEN matched_entity_ids IS NULL
                     OR trim(matched_entity_ids) = ''
                THEN []
                ELSE string_split(matched_entity_ids, ',')
            END
        ) AS t(matched_id)

        WHERE trim(matched_id) <> ''
    """)

    truth_count = con.execute("""
        SELECT COUNT(*)
        FROM truth_pairs
    """).fetchone()[0]

    print(f"Ground-truth positive pairs: {truth_count:,}")
    print(f"Time: {time.time() - start:.2f}s")

    # --------------------------------------------------------
    # EXPERIMENT 1
    # EXACT NORMALIZED NAME
    # --------------------------------------------------------

    print("\n[5/6] Experiment 1: exact normalized name")

    start = time.time()

    con.execute("""
        CREATE OR REPLACE TABLE name_candidates AS

        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS candidate_entity_id

        FROM source1 s1

        INNER JOIN source2 s2
            ON s1.name_norm = s2.name_norm
            AND s1.name_norm <> ''

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS candidate_entity_id

        FROM source1 s1

        INNER JOIN source3 s3
            ON s1.name_norm = s3.name_norm
            AND s1.name_norm <> ''
    """)

    name_candidate_count = con.execute("""
        SELECT COUNT(*)
        FROM name_candidates
    """).fetchone()[0]

    print(f"Candidate pairs: {name_candidate_count:,}")

    # --------------------------------------------------------
    # BLOCKING RECALL
    # --------------------------------------------------------

    name_recall = con.execute("""
        SELECT
            COUNT(DISTINCT t.source1_entity_id || '|' || t.matched_entity_id)
        FROM truth_pairs t

        INNER JOIN name_candidates c

            ON t.source1_entity_id = c.source1_entity_id
            AND t.matched_entity_id = c.candidate_entity_id
    """).fetchone()[0]

    print(f"True pairs recovered: {name_recall:,}")

    print(f"""
Blocking recall:
    {name_recall:,} / {truth_count:,}
    = {name_recall / truth_count * 100:.4f}%
""")

    print(f"Time: {time.time() - start:.2f}s")

    # --------------------------------------------------------
    # EXPERIMENT 2
    # EXACT NORMALIZED NAME + COUNTRY
    # --------------------------------------------------------

    print("\n[6/6] Experiment 2: normalized name + country")

    start = time.time()

    con.execute("""
        CREATE OR REPLACE TABLE name_country_candidates AS

        SELECT
            s1.entity_id AS source1_entity_id,
            s2.entity_id AS candidate_entity_id

        FROM source1 s1

        INNER JOIN source2 s2
            ON s1.name_norm = s2.name_norm
            AND s1.country = s2.country
            AND s1.name_norm <> ''

        UNION ALL

        SELECT
            s1.entity_id AS source1_entity_id,
            s3.entity_id AS candidate_entity_id

        FROM source1 s1

        INNER JOIN source3 s3

            ON s1.name_norm = s3.name_norm
            AND s1.country = s3.country
            AND s1.name_norm <> ''
    """)

    nc_candidate_count = con.execute("""
        SELECT COUNT(*)
        FROM name_country_candidates
    """).fetchone()[0]

    nc_recall = con.execute("""
        SELECT
            COUNT(DISTINCT t.source1_entity_id || '|' || t.matched_entity_id)

        FROM truth_pairs t

        INNER JOIN name_country_candidates c

            ON t.source1_entity_id = c.source1_entity_id
            AND t.matched_entity_id = c.candidate_entity_id
    """).fetchone()[0]

    print(f"Candidate pairs: {nc_candidate_count:,}")
    print(f"True pairs recovered: {nc_recall:,}")

    print(f"""
Blocking recall:
    {nc_recall:,} / {truth_count:,}
    = {nc_recall / truth_count * 100:.4f}%
""")

    print(f"Time: {time.time() - start:.2f}s")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("BLOCKING EXPERIMENT SUMMARY")
    print("=" * 75)

    print(f"""
Ground-truth positive pairs : {truth_count:,}

Exact normalized name
    Candidates              : {name_candidate_count:,}
    True pairs recovered     : {name_recall:,}
    Recall                   : {name_recall / truth_count * 100:.4f}%

Normalized name + country
    Candidates              : {nc_candidate_count:,}
    True pairs recovered     : {nc_recall:,}
    Recall                   : {nc_recall / truth_count * 100:.4f}%

Total runtime               : {time.time() - start_total:.2f}s
""")

    print("=" * 75)
    print("EXPERIMENT COMPLETE")
    print("=" * 75)

    con.close()


if __name__ == "__main__":
    main()