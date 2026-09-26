import duckdb
import os
import time

DB_PATH = os.path.abspath("blocking_experiment.duckdb")
OUT_DIR = os.path.abspath("output")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("TEST CANDIDATE GENERATION")
print("=" * 70)
print("Database:", DB_PATH)

con = duckdb.connect(DB_PATH)

con.execute("SET memory_limit='6GB'")
con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")

start = time.time()

# ------------------------------------------------------------
# 1. Load TEST data into DuckDB
# ------------------------------------------------------------
print("\n[1/6] Loading test datasets...")

con.execute("DROP TABLE IF EXISTS test_source1")
con.execute("DROP TABLE IF EXISTS test_source2")
con.execute("DROP TABLE IF EXISTS test_source3")

con.execute("""
CREATE TABLE test_source1 AS
SELECT *
FROM read_csv_auto(
    'dataset/test/test_source1.tsv',
    delim='\\t',
    header=true,
    quote='\"',
    escape='\"'
)
""")

con.execute("""
CREATE TABLE test_source2 AS
SELECT *
FROM read_csv_auto(
    'dataset/test/test_source2.tsv',
    delim='\\t',
    header=true,
    quote='\"',
    escape='\"'
)
""")

con.execute("""
CREATE TABLE test_source3 AS
SELECT *
FROM read_csv_auto(
    'dataset/test/test_source3.tsv',
    delim='\\t',
    header=true,
    quote='\"',
    escape='\"'
)
""")

for table in ["test_source1", "test_source2", "test_source3"]:
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n:,}")

# ------------------------------------------------------------
# 2. Combine Source 2 + Source 3
# ------------------------------------------------------------
print("\n[2/6] Combining Source 2 + Source 3...")

con.execute("DROP TABLE IF EXISTS test_source23")

con.execute("""
CREATE TABLE test_source23 AS
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
""")

print(
    "Combined Source 2+3:",
    f"{con.execute('SELECT COUNT(*) FROM test_source23').fetchone()[0]:,}"
)

# ------------------------------------------------------------
# 3. Normalize names and addresses
# ------------------------------------------------------------
print("\n[3/6] Normalizing names and addresses...")

con.execute("DROP TABLE IF EXISTS test_s1_norm")
con.execute("DROP TABLE IF EXISTS test_s23_norm")

# Keep normalization conservative.
# This removes punctuation/spacing differences while preserving
# alphanumeric information.

con.execute(r"""
CREATE TABLE test_s1_norm AS
SELECT
    entity_id,
    business_name,
    business_address,
    country,

    lower(
        regexp_replace(
            coalesce(business_name, ''),
            '[^a-zA-Z0-9]+',
            ' ',
            'g'
        )
    ) AS norm_name,

    lower(
        regexp_replace(
            coalesce(business_address, ''),
            '[^a-zA-Z0-9]+',
            ' ',
            'g'
        )
    ) AS norm_address

FROM test_source1
""")

con.execute(r"""
CREATE TABLE test_s23_norm AS
SELECT
    entity_id,
    business_name,
    business_address,
    country,

    lower(
        regexp_replace(
            coalesce(business_name, ''),
            '[^a-zA-Z0-9]+',
            ' ',
            'g'
        )
    ) AS norm_name,

    lower(
        regexp_replace(
            coalesce(business_address, ''),
            '[^a-zA-Z0-9]+',
            ' ',
            'g'
        )
    ) AS norm_address

FROM test_source23
""")

# ------------------------------------------------------------
# 4. Exact-name + exact-address candidates
# ------------------------------------------------------------
print("\n[4/6] Generating exact name/address candidates...")

con.execute("DROP TABLE IF EXISTS test_exact_candidates")

con.execute("""
CREATE TABLE test_exact_candidates AS

SELECT DISTINCT
    s1.entity_id AS source1_entity_id,
    s23.entity_id AS matched_entity_id

FROM test_s1_norm s1
JOIN test_s23_norm s23
    ON s1.country = s23.country
   AND s1.norm_name <> ''
   AND s1.norm_name = s23.norm_name

UNION

SELECT DISTINCT
    s1.entity_id AS source1_entity_id,
    s23.entity_id AS matched_entity_id

FROM test_s1_norm s1
JOIN test_s23_norm s23
    ON s1.country = s23.country
   AND s1.norm_address <> ''
   AND s1.norm_address = s23.norm_address
""")

exact_count = con.execute(
    "SELECT COUNT(*) FROM test_exact_candidates"
).fetchone()[0]

print(f"Exact candidates: {exact_count:,}")

# ------------------------------------------------------------
# 5. Token-based Top-50 retrieval
# ------------------------------------------------------------
print("\n[5/6] Building token retrieval index...")

con.execute("DROP TABLE IF EXISTS test_s23_tokens")

con.execute(r"""
CREATE TABLE test_s23_tokens AS
SELECT DISTINCT
    entity_id,
    lower(token) AS token
FROM test_s23_norm,
UNNEST(
    regexp_extract_all(
        norm_name,
        '[a-z0-9]{4,}'
    )
) AS t(token)
WHERE length(token) >= 4
""")

con.execute("""
CREATE INDEX IF NOT EXISTS idx_test_s23_tokens_token
ON test_s23_tokens(token)
""")

print(
    "S23 name tokens:",
    f"{con.execute('SELECT COUNT(*) FROM test_s23_tokens').fetchone()[0]:,}"
)

con.execute("DROP TABLE IF EXISTS test_token_df")

con.execute("""
CREATE TABLE test_token_df AS
SELECT
    token,
    COUNT(DISTINCT entity_id) AS df
FROM test_s23_tokens
GROUP BY token
""")

# Ignore extremely common tokens.
MAX_TOKEN_DF = 3000

con.execute("DROP TABLE IF EXISTS test_useful_tokens")

con.execute(f"""
CREATE TABLE test_useful_tokens AS
SELECT token
FROM test_token_df
WHERE df <= {MAX_TOKEN_DF}
""")

print(
    "Useful tokens:",
    f"{con.execute('SELECT COUNT(*) FROM test_useful_tokens').fetchone()[0]:,}"
)

con.execute("DROP TABLE IF EXISTS test_s1_tokens")

con.execute(r"""
CREATE TABLE test_s1_tokens AS
SELECT DISTINCT
    entity_id AS source1_entity_id,
    lower(token) AS token
FROM test_s1_norm,
UNNEST(
    regexp_extract_all(
        norm_name,
        '[a-z0-9]{4,}'
    )
) AS t(token)
WHERE length(token) >= 4
""")

# Keep only useful tokens.
con.execute("DROP TABLE IF EXISTS test_s1_useful_tokens")

con.execute("""
CREATE TABLE test_s1_useful_tokens AS
SELECT
    s.source1_entity_id,
    s.token
FROM test_s1_tokens s
JOIN test_useful_tokens u
    ON s.token = u.token
""")

print(
    "S1 useful token rows:",
    f"{con.execute('SELECT COUNT(*) FROM test_s1_useful_tokens').fetchone()[0]:,}"
)

# ------------------------------------------------------------
# Process S1 in hash buckets to avoid huge intermediate joins.
# ------------------------------------------------------------
print("\nGenerating Top-50 token candidates in buckets...")

con.execute("DROP TABLE IF EXISTS test_token_candidates")

con.execute("""
CREATE TABLE test_token_candidates (
    source1_entity_id VARCHAR,
    matched_entity_id VARCHAR
)
""")

BUCKETS = 64

for bucket in range(BUCKETS):

    t0 = time.time()

    con.execute("""
        INSERT INTO test_token_candidates

        WITH bucket_s1 AS (
            SELECT DISTINCT
                source1_entity_id
            FROM test_s1_useful_tokens
            WHERE
                MOD(
                    ABS(HASH(source1_entity_id)),
                    ?
                ) = ?
        ),

        token_matches AS (
            SELECT
                s1.source1_entity_id,
                s23.entity_id AS matched_entity_id,
                COUNT(*) AS shared_tokens
            FROM test_s1_useful_tokens s1
            JOIN bucket_s1 b
                ON s1.source1_entity_id = b.source1_entity_id
            JOIN test_s23_tokens s23
                ON s1.token = s23.token
            JOIN test_s1_norm a
                ON a.entity_id = s1.source1_entity_id
            JOIN test_s23_norm c
                ON c.entity_id = s23.entity_id
               AND a.country = c.country
            GROUP BY
                s1.source1_entity_id,
                s23.entity_id
        ),

        ranked AS (
            SELECT
                source1_entity_id,
                matched_entity_id,
                shared_tokens,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id
                    ORDER BY
                        shared_tokens DESC,
                        matched_entity_id
                ) AS rn
            FROM token_matches
        )

        SELECT
            source1_entity_id,
            matched_entity_id
        FROM ranked
        WHERE rn <= 50
    """, [BUCKETS, bucket])

    elapsed = time.time() - t0

    print(
        f"Bucket {bucket + 1:02d}/{BUCKETS} "
        f"completed in {elapsed:.1f}s"
    )

# ------------------------------------------------------------
# 6. Final candidate union
# ------------------------------------------------------------
print("\n[6/6] Creating final candidate set...")

con.execute("DROP TABLE IF EXISTS final_test_candidates")

con.execute("""
CREATE TABLE final_test_candidates AS

SELECT
    source1_entity_id,
    matched_entity_id
FROM test_exact_candidates

UNION

SELECT
    source1_entity_id,
    matched_entity_id
FROM test_token_candidates
""")

final_count = con.execute(
    "SELECT COUNT(*) FROM final_test_candidates"
).fetchone()[0]

print(f"\nFinal candidate pairs: {final_count:,}")

# Export candidate_pairs.tsv
output_path = os.path.join(OUT_DIR, "candidate_pairs.tsv")

con.execute(f"""
COPY (
    SELECT
        source1_entity_id,
        matched_entity_id
    FROM final_test_candidates
    ORDER BY source1_entity_id, matched_entity_id
)
TO '{output_path}'
WITH (
    FORMAT CSV,
    DELIMITER '\\t',
    HEADER TRUE
)
""")

print("\nCandidate file created:")
print(output_path)

print("\nTotal runtime:", round(time.time() - start, 2), "sec")

con.close()

print("=" * 70)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 70)
