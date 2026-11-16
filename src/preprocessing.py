"""
MODULE 1: Dataset Verification and Preprocessing
=================================================
Loads Fake.csv and True.csv, inspects, cleans, combines,
and saves a processed dataset ready for feature extraction.

IMPORTANT DATA NOTES
--------------------
- Fake.csv  : Collected fake/misleading news articles (label = 0)
- True.csv  : SYNTHETIC AI-GENERATED articles used as the "real" class (label = 1)
              These are NOT verified real-world news articles.
              They serve as the counter-class for model training only.
"""

import os
import sys
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FAKE_PATH   = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_PATH   = os.path.join(DATASET_DIR, "True.csv")
OUTPUT_PATH = os.path.join(DATASET_DIR, "processed_dataset.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP = "─" * 65

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def load_csv(path: str, label_name: str) -> pd.DataFrame:
    """Load a CSV safely, trying UTF-8 then latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="warn")
            print(f"  ✔ Loaded '{os.path.basename(path)}'  "
                  f"[{enc}]  →  {len(df):,} rows, {df.shape[1]} columns")
            return df
        except Exception as err:
            print(f"  ✘ Failed with {enc}: {err}")
    sys.exit(f"ERROR: Cannot read {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — Loading raw CSV files")

fake_raw = load_csv(FAKE_PATH, "Fake")
true_raw = load_csv(TRUE_PATH, "True")

fake_rows_initial = len(fake_raw)
true_rows_initial = len(true_raw)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Column inspection & compatibility check
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — Column Inspection")

print(f"\n  Fake.csv  columns : {list(fake_raw.columns)}")
print(f"  True.csv  columns : {list(true_raw.columns)}")

fake_cols = set(fake_raw.columns.str.strip().str.lower())
true_cols = set(true_raw.columns.str.strip().str.lower())

shared   = fake_cols & true_cols
only_fake = fake_cols - true_cols
only_true = true_cols - fake_cols

print(f"\n  Shared columns    : {sorted(shared)}")
print(f"  Only in Fake.csv  : {sorted(only_fake)  or 'None'}")
print(f"  Only in True.csv  : {sorted(only_true)  or 'None'}")

if not shared:
    sys.exit("ERROR: No shared columns — datasets are incompatible.")

# Normalise column names (strip whitespace)
fake_raw.columns = fake_raw.columns.str.strip()
true_raw.columns = true_raw.columns.str.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Missing value analysis
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Missing Value Analysis")

for name, df in [("Fake.csv", fake_raw), ("True.csv", true_raw)]:
    nulls = df.isnull().sum()
    print(f"\n  [{name}]")
    for col, n in nulls.items():
        pct = n / len(df) * 100
        flag = "  ⚠" if n > 0 else ""
        print(f"    {col:<12}: {n:>6,} missing  ({pct:.2f}%){flag}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Duplicate record analysis
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Duplicate Record Analysis")

fake_dups = fake_raw.duplicated().sum()
true_dups = true_raw.duplicated().sum()
print(f"\n  Fake.csv  duplicates : {fake_dups:,}")
print(f"  True.csv  duplicates : {true_dups:,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Assign labels & add source flag
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — Assigning Labels")

fake_df = fake_raw.copy()
true_df = true_raw.copy()

fake_df["label"]  = 0          # Fake news
true_df["label"]  = 1          # Synthetic "real" class
true_df["synthetic"] = True    # Flag: not verified real-world news
fake_df["synthetic"] = False

print("\n  Label mapping:")
print("    0  →  Fake news  (Fake.csv)")
print("    1  →  Synthetic articles  (True.csv)")
print("\n  ⚠ WARNING: True.csv contains AI-generated synthetic articles.")
print("    They are NOT verified real-world news. Label=1 represents the")
print("    counter-class for model training purposes only.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Combine & clean
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 6 — Combining Datasets")

# Keep only columns present in both + label/synthetic
keep_cols = sorted(shared) + ["label", "synthetic"]

# Align both to the intersection of columns (plus label/synthetic)
shared_named = sorted(shared)   # e.g. ['date','subject','text','title']

fake_aligned = fake_df[shared_named + ["label", "synthetic"]]
true_aligned = true_df[shared_named + ["label", "synthetic"]]

combined = pd.concat([fake_aligned, true_aligned], ignore_index=True)
rows_after_concat = len(combined)
print(f"\n  Rows after concat : {rows_after_concat:,}")


# ── 6a: Drop full duplicates ──────────────────────────────────────────────────
section("STEP 6a — Removing Duplicate Rows")

before = len(combined)
combined.drop_duplicates(inplace=True)
after  = len(combined)
removed = before - after
print(f"\n  Duplicates removed : {removed:,}")
print(f"  Rows remaining     : {after:,}")


# ── 6b: Combine title + text into 'content' ───────────────────────────────────
section("STEP 6b — Building 'content' Column  (title + text)")

titles = combined["title"].fillna("").astype(str).str.strip()
texts  = combined["text"].fillna("").astype(str).str.strip()

# Build content in chunks to avoid MemoryError on large string concatenation
CHUNK = 5000
parts = []
for start in range(0, len(combined), CHUNK):
    t = titles.iloc[start:start+CHUNK]
    x = texts.iloc[start:start+CHUNK]
    parts.append((t + " " + x).str.strip())
combined["content"] = pd.concat(parts).reset_index(drop=True)
del parts, titles, texts


# ── 6c: Drop rows where content is empty / whitespace-only ───────────────────
section("STEP 6c — Removing Empty / Invalid Content Rows")

before = len(combined)
# Use simple strip + len check to avoid regex MemoryError
empty_mask = combined["content"].str.strip().str.len() == 0
combined = combined[~empty_mask].copy()
after   = len(combined)
removed_empty = before - after
print(f"\n  Empty-content rows removed : {removed_empty:,}")
print(f"  Rows remaining             : {after:,}")


# ── 6d: Drop rows where content is implausibly short (<= 10 chars) ───────────
section("STEP 6d — Removing Implausibly Short Content Rows (<= 10 chars)")

before = len(combined)
short_mask = combined["content"].str.len() <= 10
combined = combined[~short_mask].copy()
after   = len(combined)
removed_short = before - after
print(f"\n  Too-short rows removed : {removed_short:,}")
print(f"  Rows remaining         : {after:,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Class distribution
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 7 — Class Distribution")

dist = combined["label"].value_counts().sort_index()
total = len(combined)
print(f"\n  {'Label':<8} {'Class':<30} {'Count':>8}  {'%':>6}")
print(f"  {'─'*8} {'─'*30} {'─'*8}  {'─'*6}")
label_names = {0: "Fake news (Fake.csv)", 1: "Synthetic real (True.csv)"}
for lbl, cnt in dist.items():
    print(f"  {lbl:<8} {label_names[lbl]:<30} {cnt:>8,}  {cnt/total*100:>5.1f}%")
print(f"\n  Total clean rows : {total:,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save processed dataset
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 8 — Saving Processed Dataset")

final_cols = ["title", "text", "content", "subject", "date", "label", "synthetic"]
# Only include columns that exist in combined
final_cols = [c for c in final_cols if c in combined.columns]
combined[final_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n  ✔ Saved → {OUTPUT_PATH}")
print(f"  Columns : {final_cols}")
print(f"  Rows    : {len(combined):,}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY REPORT")

print(f"""
  Source files (UNCHANGED)
  ─────────────────────────────────────────
  Fake.csv           : {fake_rows_initial:>8,} rows  (label=0, real fake-news corpus)
  True.csv           : {true_rows_initial:>8,} rows  (label=1, synthetic AI articles)

  Rows removed
  ─────────────────────────────────────────
  Duplicates removed : {removed:>8,}
  Empty content      : {removed_empty:>8,}
  Too short          : {removed_short:>8,}
  Total removed      : {removed + removed_empty + removed_short:>8,}

  Final dataset
  ─────────────────────────────────────────
  Total clean rows   : {total:>8,}
  Output file        : {OUTPUT_PATH}

  ⚠  IMPORTANT DISCLAIMER
  ─────────────────────────────────────────
  True.csv contains AI-generated synthetic articles. The label=1
  class does NOT represent verified real-world journalism.
  Any model trained on this data should be treated as a
  DEMONSTRATION / RESEARCH prototype only.
""")
