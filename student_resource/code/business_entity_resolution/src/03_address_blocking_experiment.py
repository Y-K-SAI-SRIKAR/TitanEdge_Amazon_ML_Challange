import duckdb
import time
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH")

start = time.time()

print("=" * 70)
print("ADDRESS BLOCKING + NAME/ADDRESS UNION EXPERIMENT")
print("=" * 70)

con = duckdb.connect(DB_PATH)

# =========================================================
# 1. EXISTING TABLES
# =========================================================

print("\nExisting tables:")

for table in con.execute("SHOW TABLES").fetchall():
    print("  ", table[0])


# =========================================================
# 2. ADDRESS NORMALIZATION
# =========================================================

print("\nCreating normalized address columns...")

for table in ["source1", "source2", "source3"]:

    columns = con.execute(
        f"DESCRIBE {table}"
    ).fetchall()

    column_names = [c[0] for c in columns]

    if "norm_address" not in column_names:
        con.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN norm_address VARCHAR
        """)

    con.execute(f"""
        UPDATE {table}
        SET norm_address =
            regexp_replace(
                lower(coalesce(business_address, '')),
                '[^a-z0-9]+',
                ' ',
                'g'
            )
    """)

print("Address normalization complete.")


# =========================================================
# 3. ADDRESS BLOCKING
# =========================================================

print("\nRunning exact normalized ADDRESS blocking...")

con.execute("DROP TABLE IF EXISTS address_candidates")

con.execute("""
    CREATE TABLE address_candidates AS

    SELECT
        s1.entity_id AS source1_entity_id,
        s2.entity_id AS matched_entity_id

    FROM source1 s1

    JOIN source2 s2
        ON s1.norm_address = s2.norm_address
        AND s1.norm_address <> ''
        AND s1.country = s2.country

    UNION ALL

    SELECT
        s1.entity_id AS source1_entity_id,
        s3.entity_id AS matched_entity_id

    FROM source1 s1

    JOIN source3 s3
        ON s1.norm_address = s3.norm_address
        AND s1.norm_address <> ''
        AND s1.country = s3.country
""")


address_count = con.execute("""
    SELECT COUNT(*)
    FROM address_candidates
""").fetchone()[0]


# =========================================================
# 4. TOTAL TRUE PAIRS
# =========================================================

total_true = con.execute("""
    SELECT COUNT(*)
    FROM truth_pairs
""").fetchone()[0]


# =========================================================
# 5. ADDRESS TRUE PAIRS
# =========================================================

address_true = con.execute("""
    SELECT COUNT(*)

    FROM address_candidates a

    INNER JOIN truth_pairs t

        ON a.source1_entity_id = t.source1_entity_id

        AND a.matched_entity_id = t.matched_entity_id
""").fetchone()[0]


address_recall = address_true / total_true


print(f"Address candidate pairs: {address_count:,}")
print(f"True pairs recovered:     {address_true:,}")
print(f"Address blocking recall:  {address_recall:.4%}")


# =========================================================
# 6. EXISTING NAME + COUNTRY BLOCK
# =========================================================

print("\nUsing existing NAME + COUNTRY candidate table...")

name_count = con.execute("""
    SELECT COUNT(*)
    FROM name_country_candidates
""").fetchone()[0]


# Convert candidate_entity_id -> matched_entity_id
# temporarily for easier comparison

con.execute("DROP TABLE IF EXISTS name_candidates_std")

con.execute("""
    CREATE TABLE name_candidates_std AS

    SELECT
        source1_entity_id,
        candidate_entity_id AS matched_entity_id

    FROM name_country_candidates
""")


name_true = con.execute("""
    SELECT COUNT(*)

    FROM name_candidates_std n

    INNER JOIN truth_pairs t

        ON n.source1_entity_id = t.source1_entity_id

        AND n.matched_entity_id = t.matched_entity_id
""").fetchone()[0]


name_recall = name_true / total_true


print(f"Name candidate pairs:     {name_count:,}")
print(f"Name true pairs:          {name_true:,}")
print(f"Name blocking recall:     {name_recall:.4%}")


# =========================================================
# 7. UNION
# =========================================================

print("\nCreating NAME + ADDRESS union...")

con.execute("DROP TABLE IF EXISTS union_candidates")

con.execute("""
    CREATE TABLE union_candidates AS

    SELECT
        source1_entity_id,
        matched_entity_id

    FROM name_candidates_std

    UNION

    SELECT
        source1_entity_id,
        matched_entity_id

    FROM address_candidates
""")


union_count = con.execute("""
    SELECT COUNT(*)
    FROM union_candidates
""").fetchone()[0]


union_true = con.execute("""
    SELECT COUNT(*)

    FROM union_candidates u

    INNER JOIN truth_pairs t

        ON u.source1_entity_id = t.source1_entity_id

        AND u.matched_entity_id = t.matched_entity_id
""").fetchone()[0]


union_recall = union_true / total_true


print(f"Union candidate pairs:     {union_count:,}")
print(f"Union true pairs:          {union_true:,}")
print(f"Union blocking recall:     {union_recall:.4%}")


# =========================================================
# 8. ADDRESS TRUE PAIRS NOT FOUND BY NAME
# =========================================================

new_from_address = con.execute("""
    SELECT COUNT(*)

    FROM address_candidates a

    INNER JOIN truth_pairs t

        ON a.source1_entity_id = t.source1_entity_id

        AND a.matched_entity_id = t.matched_entity_id

    WHERE NOT EXISTS (

        SELECT 1

        FROM name_candidates_std n

        WHERE n.source1_entity_id = a.source1_entity_id

        AND n.matched_entity_id = a.matched_entity_id
    )
""").fetchone()[0]


# =========================================================
# 9. OVERLAP
# =========================================================

overlap_true = con.execute("""
    SELECT COUNT(*)

    FROM name_candidates_std n

    INNER JOIN address_candidates a

        ON n.source1_entity_id = a.source1_entity_id

        AND n.matched_entity_id = a.matched_entity_id

    INNER JOIN truth_pairs t

        ON n.source1_entity_id = t.source1_entity_id

        AND n.matched_entity_id = t.matched_entity_id
""").fetchone()[0]


address_only_true = address_true - overlap_true


# =========================================================
# 10. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
TOTAL TRUE PAIRS
    {total_true:,}

NAME + COUNTRY BLOCK
    Candidates:          {name_count:,}
    True pairs:          {name_true:,}
    Recall:               {name_recall:.4%}

ADDRESS BLOCK
    Candidates:          {address_count:,}
    True pairs:          {address_true:,}
    Recall:               {address_recall:.4%}

NAME + ADDRESS UNION
    Candidates:          {union_count:,}
    True pairs:          {union_true:,}
    Recall:               {union_recall:.4%}

ADDRESS CONTRIBUTION
    New true pairs:      {new_from_address:,}
    Address-only:         {address_only_true:,}
    Found by both:        {overlap_true:,}
""")

print("=" * 70)
print(f"Runtime: {time.time() - start:.2f} seconds")
print("=" * 70)

con.close()