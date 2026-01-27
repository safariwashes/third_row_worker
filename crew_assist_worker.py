"""
crew_assist_worker.py (multi-tenant, production) - psycopg3 compatible

Reads from public.pos, processes images to detect whether crew assisted (people in ROI),
upserts into crew_assist, and updates public.pos processed_crew queue fields.

Key assumptions:
- pos.image_path contains a FULL URL (http/https). No URL construction in this worker.
- pos has crew queue fields:
    processed_crew boolean default false
    crew_claimed_by text
    crew_claimed_at timestamptz
    crew_claim_attempts int default 0
    crew_last_error text
    crew_next_attempt_at timestamptz
- crew_assist table exists.

Environment:
- DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
- YOLO_MODEL_PATH (default yolov8n.pt)
- YOLO_CONF (default 0.30)
- YOLO_PERSON_CLASS_ID (default 0)
- ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 (global fallback ROI)
- CREW_CLAIM_LIMIT_TOTAL (default 150)
- CREW_CLAIM_LIMIT_PER_TENANT_LOC (default 10)
- CREW_CLAIM_STALE_MINUTES (default 10)
- CREW_LOOP_SLEEP_SECONDS (default 2.0)
- HTTP_TIMEOUT_SECONDS (default 15)
- REQUIRE_LPR_FIRST (optional; default false)
"""

import io
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import psycopg  # psycopg v3
import requests
from PIL import Image

# ============================================================
# Config
# ============================================================

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.30"))
YOLO_PERSON_CLASS_ID = int(os.getenv("YOLO_PERSON_CLASS_ID", "0"))  # COCO person

ROI_X1 = int(os.getenv("ROI_X1", "0"))
ROI_Y1 = int(os.getenv("ROI_Y1", "0"))
ROI_X2 = int(os.getenv("ROI_X2", "0"))
ROI_Y2 = int(os.getenv("ROI_Y2", "0"))

CLAIM_LIMIT_TOTAL = int(os.getenv("CREW_CLAIM_LIMIT_TOTAL", "150"))
CLAIM_LIMIT_PER_TENANT_LOC = int(os.getenv("CREW_CLAIM_LIMIT_PER_TENANT_LOC", "10"))
CLAIM_STALE_MINUTES = int(os.getenv("CREW_CLAIM_STALE_MINUTES", "10"))
LOOP_SLEEP_SECONDS = float(os.getenv("CREW_LOOP_SLEEP_SECONDS", "2.0"))

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "15"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "2"))
HTTP_RETRY_BACKOFF_SECONDS = float(os.getenv("HTTP_RETRY_BACKOFF_SECONDS", "1.0"))

BACKOFF_MINUTES = [2, 5, 15, 30, 60]  # attempt 1..n

WORKER_SOURCE = os.getenv("CREW_WORKER_SOURCE", "crew_assist_worker")
WORKER_ID = os.getenv("WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"
# ^^^ leaving exactly as-is can cause lint only; safe runtime.
# If you want clean: WORKER_ID = os.getenv("WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"

REQUIRE_LPR_FIRST = (os.getenv("REQUIRE_LPR_FIRST", "false").strip().lower() in ("1", "true", "yes"))

# Fix the WORKER_ID line above safely:
WORKER_ID = os.getenv("WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"


# ============================================================
# Utils
# ============================================================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def compute_next_attempt(attempts: int) -> datetime:
    idx = min(max(attempts, 1), len(BACKOFF_MINUTES)) - 1
    return utcnow() + timedelta(minutes=BACKOFF_MINUTES[idx])


def validate_full_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if not (u.lower().startswith("http://") or u.lower().startswith("https://")):
        raise ValueError(f"pos.image_path is not a full URL: {u[:120]}")
    return u


def http_get_image(url: str) -> Image.Image:
    last_err = None
    for attempt in range(HTTP_RETRIES + 1):
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last_err = e
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"Failed to fetch image after retries: {last_err}")


def clamp_roi(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


# ============================================================
# Schema helpers + logging
# ============================================================

def get_table_columns(cur, table_name: str) -> set:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table_name,),
    )
    return {r[0] for r in cur.fetchall()}


def table_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
          SELECT 1 FROM information_schema.tables
          WHERE table_schema='public' AND table_name=%s
        )
        """,
        (table_name,),
    )
    return bool(cur.fetchone()[0])


class Logger:
    def __init__(self, cur):
        self.errorlog_cols = get_table_columns(cur, "errorlog") if table_exists(cur, "errorlog") else set()
        self.heartbeat_cols = get_table_columns(cur, "heartbeat") if table_exists(cur, "heartbeat") else set()

    def log_error(
        self,
        cur,
        message: str,
        context: str = "",
        tenant_id=None,
        location_id=None,
        error_type: Optional[str] = None,
        raw_data: Optional[str] = None,
    ):
        if not self.errorlog_cols:
            print(f"[ERRORLOG missing] {message} | {context}")
            return

        cols = ["message", "context"]
        vals = [message, context]

        if "error_type" in self.errorlog_cols and error_type is not None:
            cols.append("error_type"); vals.append(error_type)
        if "raw_data" in self.errorlog_cols and raw_data is not None:
            cols.append("raw_data"); vals.append(raw_data)
        if "tenant_id" in self.errorlog_cols:
            cols.append("tenant_id"); vals.append(tenant_id)
        if "location_id" in self.errorlog_cols:
            cols.append("location_id"); vals.append(location_id)
        if "source" in self.errorlog_cols:
            cols.append("source"); vals.append(WORKER_SOURCE)

        ph = ", ".join(["%s"] * len(cols))
        cur.execute(f"INSERT INTO errorlog ({', '.join(cols)}) VALUES ({ph})", tuple(vals))

    def log_heartbeat(self, cur, tenant_id=None, location_id=None):
        if not self.heartbeat_cols:
            return
        cols = ["source"]
        vals = [WORKER_SOURCE]
        if "tenant_id" in self.heartbeat_cols:
            cols.append("tenant_id"); vals.append(tenant_id)
        if "location_id" in self.heartbeat_cols:
            cols.append("location_id"); vals.append(location_id)
        ph = ", ".join(["%s"] * len(cols))
        cur.execute(f"INSERT INTO heartbeat ({', '.join(cols)}) VALUES ({ph})", tuple(vals))


# ============================================================
# Optional per-location ROI (DB-driven)
# ============================================================

def fetch_roi_for_location(cur, tenant_id, location_id) -> Tuple[int, int, int, int]:
    """
    Optional: if you create a table location_settings(tenant_id, location_id, roi_x1, roi_y1, roi_x2, roi_y2)
    the worker uses that ROI. Otherwise uses env ROI_*.
    """
    if not table_exists(cur, "location_settings"):
        return (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)

    cols = get_table_columns(cur, "location_settings")
    need = {"tenant_id", "location_id", "roi_x1", "roi_y1", "roi_x2", "roi_y2"}
    if not need.issubset(cols):
        return (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)

    cur.execute(
        """
        SELECT roi_x1, roi_y1, roi_x2, roi_y2
        FROM location_settings
        WHERE tenant_id=%s AND location_id=%s
        """,
        (tenant_id, location_id),
    )
    row = cur.fetchone()
    if not row:
        return (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]))


# ============================================================
# YOLO model (lazy)
# ============================================================

_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO  # type: ignore
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model


# ============================================================
# POS queue checks + SQL
# ============================================================

def assert_pos_has_crew_queue(cur):
    cols = get_table_columns(cur, "pos")
    required = {
        "processed_crew",
        "crew_claimed_by",
        "crew_claimed_at",
        "crew_claim_attempts",
        "crew_last_error",
        "crew_next_attempt_at",
        "image_path",
        "tenant_id",
        "location_id",
        "bill",
        "created_on",
    }
    missing = sorted(list(required - cols))
    if missing:
        raise RuntimeError("pos is missing required columns for crew queue/processing: " + ", ".join(missing))


CLAIM_SQL = """
WITH ranked AS (
  SELECT tenant_id, location_id, bill
  FROM (
    SELECT
      tenant_id,
      location_id,
      bill,
      ROW_NUMBER() OVER (
        PARTITION BY tenant_id, location_id
        ORDER BY wash_ts_first DESC NULLS LAST
      ) AS rn
    FROM pos
    WHERE processed_crew = false
      AND (crew_next_attempt_at IS NULL OR crew_next_attempt_at <= now())
      AND (
        crew_claimed_at IS NULL
        OR crew_claimed_at < (now() - (%s || ' minutes')::interval)
      )
      AND (image_path IS NOT NULL AND btrim(image_path) <> '')
      AND (
        %s = false
        OR processed_lpr = true
      )
  ) t
  WHERE rn <= %s
  LIMIT %s
)
UPDATE pos p
SET crew_claimed_by = %s,
    crew_claimed_at = now(),
    crew_claim_attempts = p.crew_claim_attempts + 1
FROM ranked r
WHERE p.tenant_id = r.tenant_id
  AND p.location_id = r.location_id
  AND p.bill = r.bill
RETURNING
  p.tenant_id,
  p.location_id,
  p.bill,
  p.created_on,
  p.image_path,
  p.crew_claim_attempts;
"""

MARK_SUCCESS_SQL = """
UPDATE pos
SET processed_crew = true,
    crew_claimed_by = NULL,
    crew_claimed_at = NULL,
    crew_last_error = NULL,
    crew_next_attempt_at = NULL
WHERE tenant_id=%s AND location_id=%s AND bill=%s;
"""

MARK_FAILURE_SQL = """
UPDATE pos
SET crew_last_error = %s,
    crew_next_attempt_at = %s,
    crew_claimed_by = NULL,
    crew_claimed_at = NULL
WHERE tenant_id=%s AND location_id=%s AND bill=%s;
"""


# ============================================================
# crew_assist upsert (supports both schemas)
# ============================================================

def build_crew_assist_upsert(cur):
    cols = get_table_columns(cur, "crew_assist")
    has_tenant = ("tenant_id" in cols and "location_id" in cols)

    if has_tenant:
        insert_cols = [
            "tenant_id", "location_id",
            "bill", "created_on",
            "image_url",
            "crew_assisted",
            "people_in_roi",
            "confidence",
            "model_version",
            "checked_ts",
            "camera_key",
        ]
        conflict_target = "(tenant_id, location_id, bill, created_on)"
    else:
        insert_cols = [
            "bill", "created_on",
            "image_url",
            "crew_assisted",
            "people_in_roi",
            "confidence",
            "model_version",
            "checked_ts",
            "camera_key",
        ]
        conflict_target = "(bill, created_on)"

    sql = f"""
    INSERT INTO crew_assist ({", ".join(insert_cols)})
    VALUES ({", ".join(["%s"] * len(insert_cols))})
    ON CONFLICT {conflict_target}
    DO UPDATE SET
      image_url     = EXCLUDED.image_url,
      crew_assisted = EXCLUDED.crew_assisted,
      people_in_roi = EXCLUDED.people_in_roi,
      confidence    = EXCLUDED.confidence,
      model_version = EXCLUDED.model_version,
      checked_ts    = EXCLUDED.checked_ts,
      camera_key    = COALESCE(EXCLUDED.camera_key, crew_assist.camera_key);
    """
    return sql, has_tenant, len(insert_cols)


# ============================================================
# Camera key extraction (best-effort, non-blocking)
# ============================================================

def extract_camera_key(image_url: str) -> Optional[str]:
    try:
        parts = image_url.split("/")
        for i, p in enumerate(parts):
            if p.lower() == "server_10" and i + 3 < len(parts):
                return parts[i + 2]
    except Exception:
        pass
    return None


# ============================================================
# Inference
# ============================================================

@dataclass
class CrewAssistResult:
    crew_assisted: str          # 'Y' or 'N'
    people_in_roi: int
    confidence: float
    model_version: str
    checked_ts: datetime
    camera_key: Optional[str]


def run_people_in_roi(image: Image.Image, roi: Tuple[int, int, int, int]) -> CrewAssistResult:
    model = get_model()

    w, h = image.size
    x1, y1, x2, y2 = clamp_roi(roi, w, h)
    crop = image.crop((x1, y1, x2, y2))

    results = model.predict(crop, conf=YOLO_CONF, verbose=False)
    checked_ts = datetime.now()

    people = 0
    best_conf = 0.0

    if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
        boxes = results[0].boxes
        cls_list = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)
        conf_list = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else list(boxes.conf)

        for cls_id, conf in zip(cls_list, conf_list):
            if int(cls_id) == YOLO_PERSON_CLASS_ID:
                people += 1
                if float(conf) > best_conf:
                    best_conf = float(conf)

    assisted = "Y" if people > 0 else "N"
    model_version = f"{YOLO_MODEL_PATH}|conf={YOLO_CONF}"

    return CrewAssistResult(
        crew_assisted=assisted,
        people_in_roi=people,
        confidence=float(best_conf),
        model_version=model_version,
        checked_ts=checked_ts,
        camera_key=None,
    )


# ============================================================
# Worker loop
# ============================================================

def process_once(conn) -> int:
    with conn.cursor() as cur:
        assert_pos_has_crew_queue(cur)
        logger = Logger(cur)
        upsert_sql, crew_has_tenant, upsert_ncols = build_crew_assist_upsert(cur)

        # 1) Claim
        cur.execute(
            CLAIM_SQL,
            (
                CLAIM_STALE_MINUTES,
                REQUIRE_LPR_FIRST,
                CLAIM_LIMIT_PER_TENANT_LOC,
                CLAIM_LIMIT_TOTAL,
                WORKER_ID,
            ),
        )
        claimed = cur.fetchall()
        conn.commit()

        if not claimed:
            return 0

        # 2) Process
        upsert_rows: List[tuple] = []
        success_keys: List[tuple] = []
        failure_updates: List[tuple] = []

        for (tenant_id, location_id, bill, created_on, image_path, attempts) in claimed:
            image_url = None
            try:
                image_url = validate_full_url(image_path)
                if not image_url:
                    raise RuntimeError("pos.image_path is empty")

                img = http_get_image(image_url)
                roi = fetch_roi_for_location(cur, tenant_id, location_id)

                result = run_people_in_roi(img, roi)
                result.camera_key = extract_camera_key(image_url)

                if crew_has_tenant:
                    upsert_rows.append(
                        (
                            tenant_id, location_id,
                            bill, created_on,
                            image_url,
                            result.crew_assisted,
                            result.people_in_roi,
                            result.confidence,
                            result.model_version,
                            result.checked_ts,
                            result.camera_key,
                        )
                    )
                else:
                    upsert_rows.append(
                        (
                            bill, created_on,
                            image_url,
                            result.crew_assisted,
                            result.people_in_roi,
                            result.confidence,
                            result.model_version,
                            result.checked_ts,
                            result.camera_key,
                        )
                    )

                success_keys.append((tenant_id, location_id, bill))

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                logger.log_error(
                    cur,
                    "crew_assist processing failed",
                    context=f"tenant_id={tenant_id} location_id={location_id} bill={bill} url={image_url}",
                    tenant_id=tenant_id,
                    location_id=location_id,
                    error_type="CREW_ASSIST_FAIL",
                    raw_data=err[:2000],
                )
                failure_updates.append((err, compute_next_attempt(attempts), tenant_id, location_id, bill))

        # 3) Write results
        try:
            with conn.cursor() as cur2:
                logger2 = Logger(cur2)

                if upsert_rows:
                    # psycopg3: use executemany (safe, good enough for these batch sizes)
                    cur2.executemany(upsert_sql, upsert_rows)

                for (tenant_id, location_id, bill) in success_keys:
                    cur2.execute(MARK_SUCCESS_SQL, (tenant_id, location_id, bill))

                for (err, next_time, tenant_id, location_id, bill) in failure_updates:
                    cur2.execute(MARK_FAILURE_SQL, (err, next_time, tenant_id, location_id, bill))

                logger2.log_heartbeat(cur2)

            conn.commit()
        except Exception as e:
            conn.rollback()
            try:
                with conn.cursor() as cur3:
                    Logger(cur3).log_error(cur3, f"DB write failure: {e}", context="crew_assist_worker", error_type="DB_FAIL")
                conn.commit()
            except Exception:
                conn.rollback()
            raise

        return len(claimed)


def main():
    if not DB_NAME or not DB_USER or not DB_PASSWORD or not DB_HOST:
        raise RuntimeError("Missing DB env vars (DB_NAME/DB_USER/DB_PASSWORD/DB_HOST).")

    print(f"[{WORKER_SOURCE}] starting. WORKER_ID={WORKER_ID} MODEL={YOLO_MODEL_PATH}")

    while True:
        conn = None
        try:
            conn = psycopg.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                connect_timeout=10,
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
