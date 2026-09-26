import duckdb

con = duckdb.connect("blocking_experiment.duckdb")

total = con.execute("""
SELECT COUNT(*)
FROM truth_pairs t
JOIN s1_sample s
ON t.source1_entity_id = s.source1_entity_id
""").fetchone()[0]

base = con.execute("""
SELECT COUNT(*)
FROM truth_pairs t
JOIN union_candidates u
ON t.source1_entity_id = u.source1_entity_id
AND t.matched_entity_id = u.matched_entity_id
JOIN s1_sample s
ON t.source1_entity_id = s.source1_entity_id
""").fetchone()[0]

print("=" * 60)
print("TOKEN TOP-K COMPARISON")
print("=" * 60)
print(f"Sample true pairs : {total:,}")
print(f"Existing recovered: {base:,}")
print()

for k in [20, 50, 100]:

    query = f"""
    SELECT source1_entity_id, candidate_entity_id
    FROM (
        SELECT
            source1_entity_id,
            candidate_entity_id,
            ROW_NUMBER() OVER (
                PARTITION BY source1_entity_id
                ORDER BY shared_tokens DESC
            ) AS rn
        FROM fast_token_candidates
    )
    WHERE rn <= {k}
    """

    token_count = con.execute(f"""
    SELECT COUNT(*)
    FROM ({query})
    """).fetchone()[0]

    token_true = con.execute(f"""
    SELECT COUNT(*)
    FROM truth_pairs t
    JOIN ({query}) q
    ON t.source1_entity_id = q.source1_entity_id
    AND t.matched_entity_id = q.candidate_entity_id
    JOIN s1_sample s
    ON t.source1_entity_id = s.source1_entity_id
    """).fetchone()[0]

    new_true = con.execute(f"""
    SELECT COUNT(*)
    FROM truth_pairs t
    JOIN ({query}) q
    ON t.source1_entity_id = q.source1_entity_id
    AND t.matched_entity_id = q.candidate_entity_id
    JOIN s1_sample s
    ON t.source1_entity_id = s.source1_entity_id
    WHERE NOT EXISTS (
        SELECT 1
        FROM union_candidates u
        WHERE t.source1_entity_id = u.source1_entity_id
        AND t.matched_entity_id = u.matched_entity_id
    )
    """).fetchone()[0]

    combined = base + new_true
    recall = combined / total

    print(f"TOP-{k}")
    print(f"  Token candidates : {token_count:,}")
    print(f"  Token true pairs : {token_true:,}")
    print(f"  New true pairs   : {new_true:,}")
    print(f"  Combined recall  : {recall:.4%}")
    print()

print("=" * 60)

con.close()
