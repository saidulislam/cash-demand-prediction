#!/usr/bin/env python3
"""
Generate a minimal v1 branch cash-demand dataset for ML practice.

Columns:
  - branch_id, date, atm_withdrawals, otc_withdrawals, is_holiday, is_payday
  - (optional but included) cash_deposits, is_branch_open

Notes:
  - Asks for number of rows interactively (or use --rows).
  - Injects a few nulls and duplicate rows to practice cleaning.
  - Simulates realistic seasonality (Fri↑, weekend↓), paydays, month-end bump, holidays.
  - **No future dates**: generation is capped at today's date.
"""

import argparse
import sys
import random
from datetime import date, timedelta
from typing import List, Dict, Set

import numpy as np
import pandas as pd

# -------------------------
# US Bank Holiday helpers
# -------------------------

def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    days_to_add = (weekday - d.weekday()) % 7
    d = d + timedelta(days=days_to_add)
    d = d + timedelta(weeks=n-1)
    return d

def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        d = date(year+1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month+1, 1) - timedelta(days=1)
    days_back = (d.weekday() - weekday) % 7
    return d - timedelta(days=days_back)

def us_bank_holidays(year: int) -> Set[date]:
    hol = {
        date(year, 1, 1),   # New Year's Day
        date(year, 6, 19),  # Juneteenth
        date(year, 7, 4),   # Independence Day
        date(year, 11, 11), # Veterans Day
        date(year, 12, 25), # Christmas
    }
    hol.add(nth_weekday_of_month(year, 1, 0, 3))   # MLK Day
    hol.add(nth_weekday_of_month(year, 2, 0, 3))   # Presidents' Day
    hol.add(last_weekday_of_month(year, 5, 0))     # Memorial Day
    hol.add(nth_weekday_of_month(year, 9, 0, 1))   # Labor Day
    hol.add(nth_weekday_of_month(year, 10, 0, 2))  # Indigenous Peoples'/Columbus
    hol.add(nth_weekday_of_month(year, 11, 3, 4))  # Thanksgiving
    return hol

def is_payday(d: date) -> bool:
    return d.day in (1, 15)

# -------------------------
# Data generation
# -------------------------

BRANCHES = [
    {"branch_id": "B001", "city": "Columbus",   "base": 42000},
    {"branch_id": "B002", "city": "Cleveland",  "base": 52000},
    {"branch_id": "B003", "city": "Cincinnati", "base": 47000},
    {"branch_id": "B004", "city": "Toledo",     "base": 38000},
    {"branch_id": "B005", "city": "Dayton",     "base": 36000},
    {"branch_id": "B006", "city": "Akron",      "base": 34000},
    {"branch_id": "B007", "city": "Youngstown", "base": 30000},
    {"branch_id": "B008", "city": "Lorain",     "base": 29000},
]

def _max_rows_available(start: date, today: date) -> int:
    """Max rows from start..today across all branches (one row per branch per day)."""
    if start > today:
        return 0
    days = (today - start).days + 1
    return days * len(BRANCHES)

def generate_rows(n_rows: int, seed: int = 17, start: date = date(2024,1,1)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    today = date.today()

    if start > today:
        raise ValueError(f"Start date {start} is in the future relative to today {today}.")

    # Cap rows so we never emit future dates
    max_rows = _max_rows_available(start, today)
    if n_rows > max_rows:
        print(
            f"[INFO] Requested {n_rows} rows, but only {max_rows} rows exist up to {today}. "
            f"Generating {max_rows} rows.", file=sys.stderr
        )
        n_rows = max_rows

    # Weekly pattern multipliers (Mon..Sun); Fri higher, weekend lower
    weekly = np.array([1.00, 1.03, 1.05, 1.06, 1.12, 0.82, 0.72])
    weekly = weekly / weekly.mean()

    rows: List[Dict] = []
    d = start
    branch_idx = 0
    holiday_cache: Dict[int, Set[date]] = {}

    while len(rows) < n_rows:
        # Safety: never generate future dates
        if d > today:
            break

        b = BRANCHES[branch_idx]
        branch_idx = (branch_idx + 1) % len(BRANCHES)

        # advance date when cycling back to first branch
        if branch_idx == 0 and len(rows) > 0:
            d = d + timedelta(days=1)
            if d > today:
                break

        if d.year not in holiday_cache:
            holiday_cache[d.year] = us_bank_holidays(d.year)

        is_hol = d in holiday_cache[d.year]
        is_pd  = is_payday(d)

        # month-end bump on last 2 calendar days
        tomorrow = d + timedelta(days=1)
        day_after = d + timedelta(days=2)
        month_end = (tomorrow.month != d.month) or (day_after.month != d.month)
        month_end_boost = 1.10 if month_end else 1.0

        dow = d.weekday()  # Mon=0
        level = b["base"] * weekly[dow]
        payday_boost = 1.15 if is_pd else 1.0
        holiday_multiplier = 0.35 if is_hol else 1.0

        expected = level * payday_boost * month_end_boost * holiday_multiplier
        expected *= rng.normal(1.0, 0.10)

        if rng.random() < 0.006:
            expected *= rng.uniform(2.0, 3.5)

        atm_ratio = rng.uniform(0.45, 0.65)
        atm = max(0.0, expected * atm_ratio + rng.normal(0, 600))
        otc = max(0.0, expected * (1 - atm_ratio) + rng.normal(0, 600))

        cash_deposits = max(0.0, (atm + otc) * rng.uniform(0.25, 0.55) + rng.normal(0, 800))

        is_open = 1
        if is_hol and rng.random() < 0.7:
            is_open = 0
            atm, otc = 0.0, 0.0
        elif rng.random() < 0.004:
            is_open = 0
            atm, otc = 0.0, 0.0

        rows.append({
            "branch_id": b["branch_id"],
            "date": pd.Timestamp(d).date().isoformat(),
            "atm_withdrawals": float(round(atm, 2)),
            "otc_withdrawals": float(round(otc, 2)),
            "is_holiday": int(is_hol),
            "is_payday": int(is_pd),
            "cash_deposits": float(round(cash_deposits, 2)),
            "is_branch_open": int(is_open),
        })

    df = pd.DataFrame(rows)

    # Inject imperfections (still only past/current dates)
    for col in ["atm_withdrawals", "otc_withdrawals"]:
        idx = df.sample(frac=0.02, random_state=seed).index
        df.loc[idx, col] = np.nan

    idx = df.sample(frac=0.005, random_state=seed+1).index
    df.loc[idx, "is_branch_open"] = np.nan

    dupes = df.sample(frac=0.008, random_state=seed+2)
    df = pd.concat([df, dupes], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed+3).reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None, help="Number of rows to generate (will append ~0.8% duplicates).")
    parser.add_argument("--out", type=str, default="branch_cash_data.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    # interactive prompt if rows not provided
    n_rows = args.rows
    if n_rows is None:
        try:
            raw = input("How many rows do you want to generate? [e.g., 2000]: ").strip()
            n_rows = int(raw)
        except Exception:
            print("Invalid input. Please pass --rows N (e.g., --rows 2000).", file=sys.stderr)
            sys.exit(1)

    try:
        y, m, d = map(int, args.start.split("-"))
        start_dt = date(y, m, d)
    except Exception:
        print(f"Bad --start value: {args.start}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    df = generate_rows(n_rows, seed=args.seed, start=start_dt)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} rows (including ~0.8% duplicates). Saved to: {args.out}")
    print("Columns:", ", ".join(df.columns))

if __name__ == "__main__":
    main()

# python generate_branch_cash_v1.py --rows 2920 --out branch_cash_data.csv
#
#Good minimum (per branch): ~300 daily rows (~10 months).
#That’s enough to (a) warm up 28-day features, (b) do a few time-based CV folds, and (c) hold out a 60–90-day test.
#	•	Solid/production-ish (per branch): 12–24 months (365–730 rows).
#A full year captures all holidays + month-end patterns once; two years gives you repeat exposure and is noticeably more stable.
#	•	If you pool branches in one “global” model: aim for ≥ 5k–20k total rows (e.g., 50 branches × 180 days = 9,000 rows). Pooling lets each branch “borrow strength.”

