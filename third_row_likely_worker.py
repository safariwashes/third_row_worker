import os
import re
import socket
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import psycopg2

# =========================
# Env / Config
# =========================
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")

CLAIM_LIMIT_TOTAL = int(os.getenv("THIRD_ROW_CLAIM_LIMIT_TOTAL", "500"))
CLAIM_LIMIT_PER_TENANT_LOC = int(os.getenv("THIRD_ROW_CLAIM_LIMIT_PER_TENANT_LOC", "30"))
CLAIM_STALE_MINUTES = int(os.getenv("THIRD_ROW_CLAIM_STALE_MINUTES", "10"))
LOOP_SLEEP_SECONDS = float(os.getenv("THIRD_ROW_LOOP_SLEEP_SECONDS", "2.0"))

# Backoff schedule (minutes) based on attempt count (1..n)
BACKOFF_MINUTES = [2, 5, 15, 30, 60]

WORKER_SOURCE = os.getenv("THIRD_ROW_WORKER_SOURCE", "third_row_likely_worker")
WORKER_ID = os.getenv("WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"

MODEL_VERSION = os.getenv("THIRD_ROW_MODEL_VERSION", "heuristic_v1")


# =========================
# Heuristic configuration
# =========================
# Strong 3-row models (common large SUVs / 3-row crossovers / minivans)
MODEL_KEYWORDS_STRONG = [
    # Full-size / large SUVs
    "suburban", "tahoe", "yukon", "escalade", "expedition", "navigator",
    "sequoia", "armada", "qx80", "lx", "gx", "land cruiser",
    # 3-row mid-size SUVs
    "highlander", "grand highlander", "pilot", "mdx", "pathfinder",
    "telluride", "palisade", "atlas", "durango", "ascent", "traverse",
    "acadia", "enclave", "cx-9", "cx9", "cx-90", "cx90",
    "x7", "gls", "gle", "qx60", "rx 350l", "rx350l",
    # Vans
    "sienna", "odyssey", "pacifica", "carnival", "sedona", "town & country",
    "grand caravan", "caravan",
]

# Type keywords that usually imply 3-row capability (less precise)
TYPE_KEYWORDS_WEAK = ["minivan", "suv"]

# Some common two-row models that could otherwise falsely match "suv"
# (keep short; you can expand later)
MODEL_KEYWORDS_NEGATIVE = [
    "rav4", "cr-v", "crv", "rogue", "murano", "escape", "equinox",
    "tiguan", "sportage", "tucson", "cx-5", "cx5", "forester", "outback",
    "x3", "x5", "gle coupe", "glc", "model y",
]

def normalize_text(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def third_row_likely(make: Optional[str], model: Optional[str], vtype: Optional[str]) -> Tuple[str, float, str]:
    """
    Returns: (likely_yn, conf, reason)
    likely_yn: 'Y' or 'N'
    conf: 0..1
    reason: short explanation
    """
    make_n = normalize_text(make)
    model_n = normalize_text(model)
    type_n = normalize_text(vtype)

    haystack = f"{make_n} {model_n}".strip()

    # Negative overrides
    for neg in MODEL_KEYWORDS_NEGATIVE:
        if neg in haystack:
            # Still allow minivan override if type says so
            if "minivan" in type_n:
                return ("Y", 0.85, "type=minivan override")
            return ("N", 0.20, f"negative_model_match={neg}")

    # Strong model keyword match
    for kw in MODEL_KEYWORDS_STRONG:
        if kw in haystack:
            return ("Y", 0.95, f"model_match={kw}")

    # Weak type match
    if any(tk in type_n for tk in TYPE_KEYWORDS_WEAK):
        # SUV is weak because many are 2-row
        if "minivan" in type_n:
            return ("Y", 0.85, "type=minivan")
        if "suv" in type_n:
            return ("Y", 0.60, "type=suv")

    # If we have no data, mark as not likely but low confidence
    if not haystack and not type_n:
        return ("N", 0.10, "no_make_model_type")

    return ("N", 0.40, "no_match")


# =========================
# Queue helpers
# =========================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def compute_next_attempt(attempts: int) -> datetime:
    idx = min(max(attempts, 1), len(BACKOFF_MINUTES)) - 1
    return utcnow() + timedelta(minutes=BACKOFF_MINUTES[idx])

CLAIM_SQL = """
WITH ranked AS (
  SELECT tenant_id, location_id, bill, created_on
  FROM (
    SELECT
      tenant_id,
      location_id,
      bill,
      created_on,
      ROW_NUMBER() OVER (
        PARTITION BY tenant_id, location_id
        ORDER BY created_on DESC, bill DESC
      ) AS rn
    FROM vehicle
    WHERE third_row_processed = false
      AND (third_row_next_attempt_at IS NULL OR third_row_next_attempt_at <= now())
      AND (
        third_row_claimed_at IS NULL
        OR third_row_claimed_at < (now() - (%s || ' minutes')::interval)
      )
  ) t
  WHERE rn <= %s
  LIMIT %s
)
UPDATE vehicle v
SET third_row_claimed_by = %s,
    third_row_claimed_at = now(),
    third_row_claim_attempts = v.third_row_claim_attempts + 1
FROM ranked r
WHERE v.tenant_id = r.tenant_id
  AND v.location_id = r.location_id
  AND v.bill = r.bill
  AND v.created_on = r.created_on
RETURNING
  v.tenant_id, v.location_id, v.bill, v.created_on,
  v.make, v.model, v.type,
  v.third_row_claim_attempts;
"""

MARK_SUCCESS_SQL = """
UPDATE vehicle
SET third_row_processed = true,
    third_row_claimed_by = NULL,
    third_row_claimed_at = NULL,
    third_row_last_error = NULL,
    third_row_next_attempt_at = NULL,
    third_row_likely_yn = %s,
    third_row_likely_conf = %s,
    third_row_likely_reason = %s,
    third_row_checked_ts = now(),
    third_row_model_version = %s
WHERE tenant_id=%s AND location_id=%s AND bill=%s AND created_on=%s;
"""

MARK_FAILURE_SQL = """
UPDATE vehicle
SET third_row_last_error = %s,
    third_row_next_attempt_at = %s,
    third_row_claimed_by = NULL,
    third_row_claimed_at = NULL
WHERE tenant_id=%s AND location_id=%s AND bill=%s AND created_on=%s;
"""

HEARTBEAT_SQL = "INSERT INTO heartbeat (source) VALUES (%s);"
ERRORLOG_SQL = "INSERT INTO errorlog (message, context, error_type, raw_data) VALUES (%s, %s, %s, %s);"


def process_once(conn) -> int:
    with conn.cursor() as cur:
        # 1) Claim
        cur.execute(
            CLAIM_SQL,
            (CLAIM_STALE_MINUTES, CLAIM_LIMIT_PER_TENANT_LOC, CLAIM_LIMIT_TOTAL, WORKER_ID),
        )
        rows = cur.fetchall()
        conn.commit()

        if not rows:
            return 0

        # 2) Process
        success_updates = []
        failure_updates = []

        for (tenant_id, location_id, bill, created_on, make, model, vtype, attempts) in rows:
            try:
                yn, conf, reason = third_row_likely(make, model, vtype)
                success_updates.append(
                    (yn, conf, reason, MODEL_VERSION, tenant_id, location_id, bill, created_on)
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                next_time = compute_next_attempt(attempts)
                failure_updates.append((err, next_time, tenant_id, location_id, bill, created_on))

        # 3) Write
        with conn.cursor() as cur2:
            for u in success_updates:
                cur2.execute(MARK_SUCCESS_SQL, u)

            for f in failure_updates:
                cur2.execute(MARK_FAILURE_SQL, f)

            # best-effort logs
            try:
                cur2.execute(HEARTBEAT_SQL, (WORKER_SOURCE,))
            except Exception:
                pass

            for (err, _next_time, tenant_id, location_id, bill, created_on) in failure_updates:
                try:
                    cur2.execute(
                        ERRORLOG_SQL,
                        (
                            "third_row_likely failed",
                            f"tenant_id={tenant_id} location_id={location_id} bill={bill} created_on={created_on}",
                            "THIRD_ROW_FAIL",
                            err[:2000],
                        ),
                    )
                except Exception:
                    pass

        conn.commit()
        return len(rows)


def main():
    if not (DB_NAME and DB_USER and DB_PASSWORD and DB_HOST):
        raise RuntimeError("Missing DB env vars (DB_NAME/DB_USER/DB_PASSWORD/DB_HOST).")

    print(f"[{WORKER_SOURCE}] starting. WORKER_ID={WORKER_ID} MODEL={MODEL_VERSION}")

    while True:
        conn = None
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                sslmode="require",
                connect_timeout=8,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=3,
            )
            conn.autocommit = False

            while True:
                n = process_once(conn)
                if n == 0:
                    time.sleep(LOOP_SLEEP_SECONDS)

        except Exception as e:
            print(f"[{WORKER_SOURCE}] exception: {e}")
            time.sleep(max(2.0, LOOP_SLEEP_SECONDS))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
