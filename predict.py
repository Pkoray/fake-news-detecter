"""
history.py
----------
SQLite tabanlı analiz geçmişi yönetimi.
Dış bağımlılık yok; yalnızca Python standart kütüphanesi kullanılır.
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
from datetime import datetime


# Varsayılan veritabanı yolu
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # proje kökü
    "data",
    "history.db",
)


# ─── Veritabanı kurulumu ──────────────────────────────────────────────────────

def _get_conn(db_path: str = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """SQLite bağlantısı döner; veritabanı yoksa oluşturur."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Sütun ismiyle erişim
    return conn


def init_db(db_path: str = _DEFAULT_DB_PATH) -> None:
    """
    Gerekli tabloyu oluşturur (yoksa).
    Uygulama başlarken bir kez çağrılmalıdır.
    """
    conn = _get_conn(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT    NOT NULL,
                source_type   TEXT    NOT NULL DEFAULT 'text',   -- 'text' | 'url'
                url           TEXT,
                domain        TEXT,
                text_snippet  TEXT    NOT NULL,
                full_text     TEXT,
                result        TEXT    NOT NULL,   -- 'FAKE' | 'REAL'
                confidence    REAL    NOT NULL,   -- 0.0–100.0
                risk_level    TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


# ─── Kaydetme ────────────────────────────────────────────────────────────────

def save_analysis(
    text: str,
    result: str,
    confidence: float,
    risk_level: str = "",
    url: str = "",
    domain: str = "",
    source_type: str = "text",
    db_path: str = _DEFAULT_DB_PATH,
    snippet_len: int = 200,
) -> int:
    """
    Bir analizi veritabanına kaydeder.

    Parameters
    ----------
    text : str
        Analiz edilen metin.
    result : str
        'FAKE' veya 'REAL'.
    confidence : float
        0.0–100.0 güven skoru.
    risk_level : str
        Risk seviyesi etiketi.
    url : str
        Eğer URL'den geldiyse kaynak URL.
    domain : str
        Kaynak domain.
    source_type : str
        'text' veya 'url'.
    db_path : str
        Veritabanı dosya yolu.
    snippet_len : int
        Özet için metin kısaltma uzunluğu.

    Returns
    -------
    int
        Yeni kaydın ID'si.
    """
    init_db(db_path)

    snippet = text[:snippet_len] + ("..." if len(text) > snippet_len else "")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = _get_conn(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO analyses
                (created_at, source_type, url, domain, text_snippet, full_text,
                 result, confidence, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, source_type, url, domain, snippet, text,
             result, confidence, risk_level),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


# ─── Sorgulama ───────────────────────────────────────────────────────────────

def get_history(
    filter_result: str = "Tümü",   # 'Tümü' | 'FAKE' | 'REAL'
    search_query: str = "",
    limit: int = 100,
    db_path: str = _DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Geçmiş analizleri döner.

    Parameters
    ----------
    filter_result : str
        'Tümü', 'FAKE' veya 'REAL'.
    search_query : str
        Metin/URL içinde arama yapılır.
    limit : int
        Maksimum kayıt sayısı.
    db_path : str
        Veritabanı dosya yolu.

    Returns
    -------
    list[dict]
        Her analiz bir dict olarak döner.
    """
    init_db(db_path)

    conditions = []
    params: list = []

    if filter_result in ("FAKE", "REAL"):
        conditions.append("result = ?")
        params.append(filter_result)

    if search_query.strip():
        conditions.append("(text_snippet LIKE ? OR url LIKE ? OR domain LIKE ?)")
        q = f"%{search_query.strip()}%"
        params.extend([q, q, q])

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = _get_conn(db_path)
    try:
        rows = conn.execute(
            f"""
            SELECT id, created_at, source_type, url, domain,
                   text_snippet, result, confidence, risk_level
            FROM analyses
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_stats(db_path: str = _DEFAULT_DB_PATH) -> dict:
    """
    Genel istatistikleri döner.

    Returns
    -------
    dict
        {
            "total": int,
            "fake_count": int,
            "real_count": int,
            "fake_pct": float,
            "real_pct": float,
            "avg_confidence": float,
        }
    """
    init_db(db_path)

    conn = _get_conn(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                           AS total,
                SUM(CASE WHEN result='FAKE' THEN 1 ELSE 0 END)    AS fake_count,
                SUM(CASE WHEN result='REAL' THEN 1 ELSE 0 END)    AS real_count,
                AVG(confidence)                                    AS avg_confidence
            FROM analyses
            """
        ).fetchone()

        total   = row["total"] or 0
        fake    = row["fake_count"] or 0
        real    = row["real_count"] or 0
        avg_c   = row["avg_confidence"] or 0.0

        return {
            "total":          total,
            "fake_count":     fake,
            "real_count":     real,
            "fake_pct":       round(fake / total * 100, 1) if total else 0.0,
            "real_pct":       round(real / total * 100, 1) if total else 0.0,
            "avg_confidence": round(avg_c, 1),
        }
    finally:
        conn.close()


def delete_record(record_id: int, db_path: str = _DEFAULT_DB_PATH) -> bool:
    """Belirli bir kaydı siler. Başarılıysa True döner."""
    conn = _get_conn(db_path)
    try:
        conn.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def clear_all(db_path: str = _DEFAULT_DB_PATH) -> None:
    """Tüm geçmişi temizler."""
    conn = _get_conn(db_path)
    try:
        conn.execute("DELETE FROM analyses")
        conn.commit()
    finally:
        conn.close()


# ─── CSV Dışa Aktarma ─────────────────────────────────────────────────────────

def export_to_csv(
    filter_result: str = "Tümü",
    search_query: str = "",
    db_path: str = _DEFAULT_DB_PATH,
) -> str:
    """
    Geçmiş analizleri CSV formatında string olarak döner.
    Streamlit'in st.download_button() ile kullanmak için uygundur.

    Returns
    -------
    str
        CSV içeriği.
    """
    rows = get_history(
        filter_result=filter_result,
        search_query=search_query,
        limit=10_000,
        db_path=db_path,
    )

    if not rows:
        return "id,tarih,kaynak_tipi,url,domain,metin_ozeti,sonuc,guven_skoru,risk_seviyesi\n"

    output = io.StringIO()
    fieldnames = ["id", "created_at", "source_type", "url", "domain",
                  "text_snippet", "result", "confidence", "risk_level"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()
