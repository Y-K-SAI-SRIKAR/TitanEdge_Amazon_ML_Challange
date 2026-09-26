"""
Stage 1 inference feature generation for final test candidates.

Purpose:
    Build cheap/high-value features for the 54M final test candidates
    before running the expensive XGBoost inference.

Input:
    DuckDB:
        final_test_candidates
        test_source1
        test_source2
        test_source3
        test_exact_candidates
        inspect_test_candidate_tokens

Output:
    DuckDB:
        final_inference_features

The script does NOT delete or modify candidate_pairs.tsv.
"""

import os
import time
import duckdb


# ============================================================
# CONFIG
# ============================================================

DB_PATH = os.getenv("DB_PATH", "blocking_experiment.duckdb")

MEMORY_LIMIT = "6GB"
THREADS = 4


# ============================================================
# CONNECT
# ============================================================

print("=" * 70)
print("BUILDING FINAL INFERENCE FEATURES")
print("=" * 70)

print(f"Database: {DB_PATH}")

con = duckdb.connect(DB_PATH)

con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
con.execute(f"SET threads={THREADS}")
con.execute("SET preserve_insertion_order=false")

start = time.time()


# ============================================================
# CHECK REQUIRED TABLES
# ============================================================

required_tables = [
    "final_test_candidates",
    "test_source1",
    "test_source2",
    "test_source3",
    "test_exact_candidates",
    "inspect_test_candidate_tokens",
]

existing = {
    row[0]
    for row in con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()
}

print("\nChecking required tables...")

for table in required_tables:
    if table not in existing:
        raise RuntimeError(
            f"\nMissing required table: {table}\n"
            f"Existing tables include: {sorted(existing)}"
        )
    print(f"  OK: {table}")


# ============================================================
# DROP OLD TABLE
# ============================================================

print("\nRemoving previous inference feature table if present...")

con.execute("DROP TABLE IF EXISTS final_inference_features")


# ============================================================
# CREATE INFERENCE FEATURE TABLE
# ============================================================

print("\nBuilding inference feature table...")
print("This may take several minutes for ~54M candidates.")

con.execute(
    """
    CREATE TABLE final_inference_features AS

    SELECT
        c.source1_entity_id,
        c.matched_entity_id,

        --------------------------------------------------------
        -- Candidate-level features
        --------------------------------------------------------

        CASE
            WHEN e.source1_entity_id IS NOT NULL
            THEN 1
            ELSE 0
        END AS exact_candidate,

        COALESCE(t.shared_name_tokens, 0) AS shared_name_tokens,

        --------------------------------------------------------
        -- Source 1
        --------------------------------------------------------

        s1.business_name AS s1_name,
        s1.business_address AS s1_address,
        s1.country AS s1_country,

        --------------------------------------------------------
        -- Source 2 / Source 3
        --------------------------------------------------------

        s23.business_name AS s23_name,
        s23.business_address AS s23_address,
        s23.country AS s23_country,

        --------------------------------------------------------
        -- Cheap structural features
        --------------------------------------------------------

        CASE
            WHEN s1.country = s23.country
            THEN 1
            ELSE 0
        END AS country_match,

        CASE
            WHEN s1.business_name IS NOT NULL
             AND s23.business_name IS NOT NULL
            THEN 1
            ELSE 0
        END AS both_names_present,

        CASE
            WHEN s1.business_address IS NOT NULL
             AND s23.business_address IS NOT NULL
            THEN 1
            ELSE 0
        END AS both_addresses_present,

        CASE
            WHEN s1.business_name IS NOT NULL
             AND s23.business_name IS NOT NULL
            THEN ABS(
                LENGTH(s1.business_name)
                - LENGTH(s23.business_name)
            )
            ELSE 999
        END AS name_length_diff,

        CASE
            WHEN s1.business_address IS NOT NULL
             AND s23.business_address IS NOT NULL
            THEN ABS(
                LENGTH(s1.business_address)
                - LENGTH(s23.business_address)
            )
            ELSE 999
        END AS address_length_diff

    FROM final_test_candidates c

    INNER JOIN test_source1 s1
        ON c.source1_entity_id = s1.entity_id

    INNER JOIN (
        SELECT
            entity_id,
            business_name,
            business_address,
            country
        FROM test_source2

        UNION ALL

        SELECT
            entity_id,
            business_name,
            business_address,
            country
        FROM test_source3
    ) s23
        ON c.matched_entity_id = s23.entity_id

    LEFT JOIN test_exact_candidates e
        ON c.source1_entity_id = e.source1_entity_id
       AND c.matched_entity_id = e.matched_entity_id

    LEFT JOIN inspect_test_candidate_tokens t
        ON c.source1_entity_id = t.source1_entity_id
       AND c.matched_entity_id = t.matched_entity_id
    """
)


# ============================================================
# INDEX / ORDER
# ============================================================

print("\nCreating useful indexes...")

con.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_inference_s1
    ON final_inference_features(source1_entity_id)
    """
)

con.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_inference_candidate
    ON final_inference_features(matched_entity_id)
    """
)


# ============================================================
# SUMMARY
# ============================================================

total = con.execute(
    """
    SELECT COUNT(*)
    FROM final_inference_features
    """
).fetchone()[0]

print(f"\nTotal inference rows: {total:,}")


print("\nCandidate distribution:")

rows = con.execute(
    """
    SELECT
        CASE
            WHEN shared_name_tokens = 0 THEN '0 tokens'
            WHEN shared_name_tokens = 1 THEN '1 token'
            WHEN shared_name_tokens = 2 THEN '2 tokens'
            WHEN shared_name_tokens = 3 THEN '3 tokens'
            WHEN shared_name_tokens = 4 THEN '4 tokens'
            ELSE '5+ tokens'
        END AS token_group,
        COUNT(*) AS candidate_count
    FROM final_inference_features
    GROUP BY token_group
    ORDER BY
        CASE token_group
            WHEN '0 tokens' THEN 0
            WHEN '1 token' THEN 1
            WHEN '2 tokens' THEN 2
            WHEN '3 tokens' THEN 3
            WHEN '4 tokens' THEN 4
            ELSE 5
        END
    """
).fetchall()

for group, count in rows:
    print(f"  {group:<12}: {count:,}")


print("\nExact candidate statistics:")

exact_stats = con.execute(
    """
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN exact_candidate = 1 THEN 1 ELSE 0 END) AS exact_count,
        SUM(CASE WHEN exact_candidate = 0 THEN 1 ELSE 0 END) AS non_exact_count
    FROM final_inference_features
    """
).fetchone()

print(f"  Total       : {exact_stats[0]:,}")
print(f"  Exact       : {exact_stats[1]:,}")
print(f"  Non-exact   : {exact_stats[2]:,}")


print("\nCountry statistics:")

country_stats = con.execute(
    """
    SELECT
        SUM(CASE WHEN country_match = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN country_match = 0 THEN 1 ELSE 0 END)
    FROM final_inference_features
    """
).fetchone()

print(f"  Country match     : {country_stats[0]:,}")
print(f"  Country mismatch  : {country_stats[1]:,}")


elapsed = time.time() - start

print("\n" + "=" * 70)
print("INFERENCE FEATURE BUILD COMPLETE")
print("=" * 70)
print(f"Rows:    {total:,}")
print(f"Runtime: {elapsed:.2f} seconds")
print("=" * 70)

con.close()