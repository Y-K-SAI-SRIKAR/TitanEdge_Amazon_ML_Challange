import duckdb
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "blocking_experiment.duckdb"

S1_SAMPLE = 100_000
TOP_K = 20
MAX_TOKEN_DF = 3000

print("=" * 60)
print("FAST TOKEN RETRIEVAL EXPERIMENT")
print("=" * 60)

con = duckdb.connect(str(DB))

start = time.time()

# ------------------------------------------------------------
# 1. Sample Source 1
# ------------------------------------------------------------

print("\n[1/6] Sampling Source 1...")

con.execute("DROP TABLE IF EXISTS s1_sample")

con.execute(f"""
CREATE TABLE s1_sample AS
SELECT
    entity_id AS source1_entity_id,
    business_name,
    country
FROM source1
USING SAMPLE reservoir({S1_SAMPLE} ROWS) REPEATABLE(42)
""")

s1_count = con.execute(
    "SELECT COUNT(*) FROM s1_sample"
).fetchone()[0]

print(f"S1 sample: {s1_count:,}")


# ------------------------------------------------------------
# 2. Build compact Source 2/3 name table
# ------------------------------------------------------------

print("\n[2/6] Preparing Source 2 + Source 3 names...")

con.execute("DROP TABLE IF EXISTS s23_names")

con.execute("""
CREATE TABLE s23_names AS

SELECT
    entity_id,
    business_name,
    country
FROM source2

UNION ALL

SELECT
    entity_id,
    business_name,
    country
FROM source3
""")

print(
    "S2+S3:",
    f"{con.execute('SELECT COUNT(*) FROM s23_names').fetchone()[0]:,}"
)


# ------------------------------------------------------------
# 3. Create token tables
# ------------------------------------------------------------

print("\n[3/6] Creating token tables...")

con.execute("DROP TABLE IF EXISTS s23_tokens")

con.execute("""
CREATE TABLE s23_tokens AS
SELECT DISTINCT
    entity_id,
    country,
    lower(token) AS token
FROM s23_names,
UNNEST(
    regexp_split_to_array(
        regexp_replace(
            coalesce(business_name, ''),
            '[^a-zA-Z0-9]+',
            ' ',
            'g'
        ),
        ' '
    )
) AS x(token)
WHERE length(token) >= 4
""")

# Token frequency

con.execute("DROP TABLE IF EXISTS token_df")

con.execute("""
CREATE TABLE token_df AS
SELECT
    token,
    COUNT(*) AS df
FROM s23_tokens
GROUP BY token
HAVING COUNT(*) <= ?
""", [MAX_TOKEN_DF])

# Keep only useful tokens

con.execute("DROP TABLE IF EXISTS useful_s23_tokens")

con.execute("""
CREATE TABLE useful_s23_tokens AS
SELECT
    t.entity_id,
    t.country,
    t.token
FROM s23_tokens t
INNER JOIN token_df d
    ON t.token = d.token
""")

print(
    "Useful S23 token rows:",
    f"{con.execute('SELECT COUNT(*) FROM useful_s23_tokens').fetchone()[0]:,}"
)


# ------------------------------------------------------------
# 4. Tokenize sampled S1
# ------------------------------------------------------------

print("\n[4/6] Tokenizing sampled Source 1...")

con.execute("DROP TABLE IF EXISTS s1_tokens")

con.execute("""
CREATE TABLE s1_tokens AS
SELECT DISTINCT
    source1_entity_id,
    country,
    lower(token) AS token
FROM s1_sample,
UNNEST(
    regexp_split_to_array(
        regexp_replace(
            coalesce(business_name, ''),
            '[^a-zA-Z0-9]+',
            ' ',
            'g'
        ),
        ' '
    )
) AS x(token)
WHERE length(token) >= 4
""")

con.execute("DROP TABLE IF EXISTS useful_s1_tokens")

con.execute("""
CREATE TABLE useful_s1_tokens AS
SELECT
    s.source1_entity_id,
    s.country,
    s.token
FROM s1_tokens s
INNER JOIN token_df d
    ON s.token = d.token
""")


# ------------------------------------------------------------
# 5. Retrieve candidates
# ------------------------------------------------------------

print("\n[5/6] Retrieving candidates...")

con.execute("DROP TABLE IF EXISTS fast_token_candidates")

t0 = time.time()

con.execute("""
CREATE TABLE fast_token_candidates AS

SELECT
    s.source1_entity_id,
    t.entity_id AS candidate_entity_id,
    COUNT(*) AS shared_tokens

FROM useful_s1_tokens s

INNER JOIN useful_s23_tokens t
    ON s.token = t.token
   AND s.country = t.country

GROUP BY
    s.source1_entity_id,
    t.entity_id
""")

print(
    "Raw candidates:",
    f"{con.execute('SELECT COUNT(*) FROM fast_token_candidates').fetchone()[0]:,}"
)

# Top K per S1

con.execute("DROP TABLE IF EXISTS fast_token_topk")

con.execute(f"""
CREATE TABLE fast_token_topk AS

SELECT
    source1_entity_id,
    candidate_entity_id,
    shared_tokens

FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY source1_entity_id
            ORDER BY shared_tokens DESC
        ) AS rn

    FROM fast_token_candidates
)

WHERE rn <= {TOP_K}
""")

candidate_count = con.execute("""
SELECT COUNT(*)
FROM fast_token_topk
""").fetchone()[0]

print(f"Top-K candidates: {candidate_count:,}")

print(
    f"Retrieval time: {time.time() - t0:.2f} sec"
)


# ------------------------------------------------------------
# 6. Evaluate recall
# ------------------------------------------------------------

print("\n[6/6] Evaluating recall...")

true_total = con.execute("""
SELECT COUNT(*)
FROM truth_pairs t
INNER JOIN s1_sample s
    ON t.source1_entity_id = s.source1_entity_id
""").fetchone()[0]

true_found = con.execute("""
SELECT COUNT(*)
FROM truth_pairs t
INNER JOIN fast_token_topk c
    ON t.source1_entity_id = c.source1_entity_id
   AND t.matched_entity_id = c.candidate_entity_id
""").fetchone()[0]

recall = true_found / true_total if true_total else 0

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(f"S1 sample              : {s1_count:,}")
print(f"Total true pairs       : {true_total:,}")
print(f"Recovered true pairs   : {true_found:,}")
print(f"Candidate pairs        : {candidate_count:,}")
print(f"Candidate recall       : {recall:.4%}")
print(f"Top-K                  : {TOP_K}")
print(f"Total runtime          : {time.time() - start:.2f} sec")

print("=" * 60)

con.close()