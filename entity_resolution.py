"""
entity_resolution.py — Cross-modal identity fusion.

Traditional surveillance produces FRAGMENTED identities — the same person seen
on camera A at 8am and camera B at 8:15am gets two separate profile IDs because
neither face matched cleanly. PALANTIR-grade systems FUSE these fragments by
combining multiple weak signals into a single confident identity.

This engine fuses:
    • Face feature vectors  (face_intel, 80-dim)
    • Gait signatures       (gait_engine, 18-dim)
    • Visual appearance     (color histogram, clothing, height ratio)
    • Spatial-temporal       (camera A→B transit time consistency)
    • Existing profile IDs   (visitor_profiler)

Each candidate match is scored across all available modalities, weighted by
confidence, and the result is either:
    A. Reinforce existing entity        (>0.85 fused score)
    B. Suggest merge of two entities    (0.70-0.85, written to merge_log)
    C. Spawn new entity                  (<0.70)

Engine maintains an ENTITY GRAPH with:
    • Nodes:  resolved entities (with confidence)
    • Edges:  evidence trail (which modalities matched, when, score)

The merge_log is human-reviewable so wrong merges can be undone with
unmerge_entity(eid). Merges propagate transitively — merging A→B and later
B→C means A,B,C are now one entity.

CLAUDE-CODE EXTENSION POINTS:
    - Add a modality: implement _score_<modality>() and append to FUSION_WEIGHTS
    - Adjust thresholds: edit MERGE_THRESHOLD / NEW_THRESHOLD constants
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

# ── Storage ──────────────────────────────────────────────────────

def _resolve_db_path() -> Path:
    env = os.environ.get("ENTITY_RESOLUTION_DB")
    if env:
        p = Path(env)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            pass
    for cand in [Path("/data/palm_entities.db"), Path("/tmp/palm_entities.db")]:
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            with open(cand, "ab"):
                pass
            return cand
        except OSError:
            continue
    return Path("/tmp/palm_entities.db")


DB_PATH = _resolve_db_path()
_DB_LOCK = threading.Lock()


# ── Fusion configuration ────────────────────────────────────────

FUSION_WEIGHTS: dict[str, float] = {
    "face":       0.40,
    "gait":       0.30,
    "appearance": 0.15,
    "spatial":    0.10,
    "profile":    0.05,
}

MERGE_THRESHOLD    = 0.85
SUGGEST_THRESHOLD  = 0.70
TRANSIT_MAX_SEC    = 600   # 10min — max plausible time between adjacent cameras


# ── DB schema ────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _DB_LOCK, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id          TEXT PRIMARY KEY,
            label       TEXT,
            created_ts  REAL NOT NULL,
            updated_ts  REAL NOT NULL,
            confidence  REAL DEFAULT 0.5,
            face_vec    TEXT,
            gait_vec    TEXT,
            appearance  TEXT,
            sighting_count INTEGER DEFAULT 0,
            merged_from TEXT
        );
        CREATE TABLE IF NOT EXISTS entity_aliases (
            alias_id    TEXT PRIMARY KEY,
            entity_id   TEXT NOT NULL,
            merged_ts   REAL NOT NULL,
            score       REAL,
            evidence    TEXT
        );
        CREATE TABLE IF NOT EXISTS sightings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id   TEXT NOT NULL,
            camera      TEXT NOT NULL,
            ts          REAL NOT NULL,
            modalities  TEXT,
            fused_score REAL
        );
        CREATE TABLE IF NOT EXISTS merge_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL NOT NULL,
            from_id     TEXT NOT NULL,
            into_id     TEXT NOT NULL,
            score       REAL,
            modalities  TEXT,
            reviewed    INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_sight_ent ON sightings(entity_id);
        CREATE INDEX IF NOT EXISTS idx_sight_ts  ON sightings(ts);
        CREATE INDEX IF NOT EXISTS idx_alias_ent ON entity_aliases(entity_id);
        """)


# ── Vector ops ───────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, dot / (na * nb))


def _appearance_sim(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    score, n = 0.0, 0
    for key in ("clothing_color", "height_ratio", "build"):
        if key in a and key in b:
            n += 1
            if isinstance(a[key], (int, float)):
                diff = abs(a[key] - b[key])
                score += max(0.0, 1.0 - diff)
            else:
                score += 1.0 if a[key] == b[key] else 0.0
    return score / n if n else 0.0


# ── Fusion engine ────────────────────────────────────────────────

class EntityResolver:
    def __init__(self) -> None:
        init_db()
        print(f"[entity_res] EntityResolver initialized → {DB_PATH}", flush=True)

    # ── Sightings ──
    def observe(self, *, profile_id: str | None = None, camera: str = "",
                ts: float | None = None,
                face_vec: list[float] | None = None,
                gait_vec: list[float] | None = None,
                appearance: dict | None = None) -> dict:
        """Record a sighting and resolve it to an entity (creating/merging as needed)."""
        ts = ts or time.time()
        candidates = self._score_candidates(face_vec, gait_vec, appearance, profile_id, camera, ts)
        if candidates and candidates[0]["score"] >= MERGE_THRESHOLD:
            ent = candidates[0]
            self._reinforce_entity(ent["id"], face_vec, gait_vec, appearance, ts)
            self._record_sighting(ent["id"], camera, ts, ent["modalities"], ent["score"])
            return {"action": "reinforce", "entity_id": ent["id"],
                    "score": ent["score"], "modalities": ent["modalities"]}

        if candidates and candidates[0]["score"] >= SUGGEST_THRESHOLD:
            # spawn-but-flag-for-merge
            new_id = self._spawn_entity(face_vec, gait_vec, appearance, ts, profile_id)
            self._log_merge_suggestion(new_id, candidates[0]["id"],
                                       candidates[0]["score"], candidates[0]["modalities"])
            self._record_sighting(new_id, camera, ts, candidates[0]["modalities"], candidates[0]["score"])
            return {"action": "spawn_with_suggest_merge", "entity_id": new_id,
                    "suggested_merge": candidates[0]["id"], "score": candidates[0]["score"]}

        new_id = self._spawn_entity(face_vec, gait_vec, appearance, ts, profile_id)
        self._record_sighting(new_id, camera, ts, [], 0.0)
        return {"action": "spawn", "entity_id": new_id, "score": 0.0}

    # ── Candidate scoring ──
    def _score_candidates(self, face_vec, gait_vec, appearance, profile_id,
                          camera: str, ts: float) -> list[dict]:
        with _DB_LOCK, _conn() as c:
            rows = c.execute("""SELECT id, face_vec, gait_vec, appearance, updated_ts
                                FROM entities ORDER BY updated_ts DESC LIMIT 200""").fetchall()
            # last-camera per entity for spatial scoring
            last_cam = dict(c.execute("""
                SELECT entity_id, MAX(ts) AS t FROM sightings GROUP BY entity_id
            """).fetchall())

        results: list[dict] = []
        for eid, fv, gv, ap, _ut in rows:
            modalities, weighted, total_w = [], 0.0, 0.0

            if face_vec and fv:
                try:
                    s = _cosine(face_vec, json.loads(fv))
                    weighted += s * FUSION_WEIGHTS["face"]
                    total_w  += FUSION_WEIGHTS["face"]
                    if s > 0.6: modalities.append(f"face:{s:.2f}")
                except Exception:
                    pass
            if gait_vec and gv:
                try:
                    s = _cosine(gait_vec, json.loads(gv))
                    weighted += s * FUSION_WEIGHTS["gait"]
                    total_w  += FUSION_WEIGHTS["gait"]
                    if s > 0.6: modalities.append(f"gait:{s:.2f}")
                except Exception:
                    pass
            if appearance and ap:
                try:
                    s = _appearance_sim(appearance, json.loads(ap))
                    weighted += s * FUSION_WEIGHTS["appearance"]
                    total_w  += FUSION_WEIGHTS["appearance"]
                    if s > 0.6: modalities.append(f"appear:{s:.2f}")
                except Exception:
                    pass
            if profile_id:
                # if we already have an alias for this profile, big boost
                with _DB_LOCK, _conn() as c2:
                    found = c2.execute("SELECT 1 FROM entity_aliases WHERE alias_id=? AND entity_id=?",
                                       (profile_id, eid)).fetchone()
                if found:
                    weighted += 1.0 * FUSION_WEIGHTS["profile"]
                    total_w  += FUSION_WEIGHTS["profile"]
                    modalities.append("profile:1.00")
            # spatial: did entity appear at *this* camera within plausible transit?
            ent_last = last_cam.get(eid)
            if ent_last and (ts - ent_last) <= TRANSIT_MAX_SEC:
                weighted += 0.8 * FUSION_WEIGHTS["spatial"]
                total_w  += FUSION_WEIGHTS["spatial"]
                modalities.append(f"spatial:{int(ts-ent_last)}s")

            if total_w == 0:
                continue
            score = weighted / total_w
            results.append({"id": eid, "score": score, "modalities": modalities})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:5]

    # ── Mutations ──
    def _spawn_entity(self, face_vec, gait_vec, appearance, ts: float,
                      profile_id: str | None) -> str:
        eid = f"E{int(ts*1000)%10**10:010d}"
        with _DB_LOCK, _conn() as c:
            c.execute("""INSERT INTO entities
                (id, label, created_ts, updated_ts, confidence, face_vec, gait_vec,
                 appearance, sighting_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (eid, profile_id or eid, ts, ts, 0.5,
                 json.dumps(face_vec) if face_vec else None,
                 json.dumps(gait_vec) if gait_vec else None,
                 json.dumps(appearance) if appearance else None))
            if profile_id:
                c.execute("""INSERT OR IGNORE INTO entity_aliases
                    (alias_id, entity_id, merged_ts, score, evidence)
                    VALUES (?, ?, ?, ?, ?)""",
                    (profile_id, eid, ts, 1.0, "spawn-link"))
        return eid

    def _reinforce_entity(self, eid: str, face_vec, gait_vec, appearance, ts: float) -> None:
        with _DB_LOCK, _conn() as c:
            row = c.execute("""SELECT face_vec, gait_vec, appearance, sighting_count, confidence
                               FROM entities WHERE id=?""", (eid,)).fetchone()
            if not row:
                return
            ofv, ogv, oap, sc, conf = row
            sc = (sc or 0) + 1
            conf = min(0.99, (conf or 0.5) + 0.02)

            new_fv = self._ema(face_vec, ofv) if face_vec else ofv
            new_gv = self._ema(gait_vec, ogv) if gait_vec else ogv
            new_ap = json.dumps(appearance) if appearance else oap

            c.execute("""UPDATE entities SET updated_ts=?, sighting_count=?, confidence=?,
                         face_vec=?, gait_vec=?, appearance=? WHERE id=?""",
                      (ts, sc, conf, new_fv, new_gv, new_ap, eid))

    def _ema(self, new_vec: list[float], old_json: str | None, alpha: float = 0.3) -> str:
        if not old_json:
            return json.dumps(new_vec)
        try:
            old = json.loads(old_json)
            if len(old) != len(new_vec):
                return json.dumps(new_vec)
            blended = [(1 - alpha) * o + alpha * n for o, n in zip(old, new_vec)]
            return json.dumps(blended)
        except Exception:
            return json.dumps(new_vec)

    def _record_sighting(self, eid: str, camera: str, ts: float,
                         modalities: list[str], score: float) -> None:
        with _DB_LOCK, _conn() as c:
            c.execute("""INSERT INTO sightings (entity_id, camera, ts, modalities, fused_score)
                         VALUES (?, ?, ?, ?, ?)""",
                      (eid, camera, ts, json.dumps(modalities), score))

    def _log_merge_suggestion(self, from_id: str, into_id: str,
                              score: float, modalities: list[str]) -> None:
        with _DB_LOCK, _conn() as c:
            c.execute("""INSERT INTO merge_log (ts, from_id, into_id, score, modalities)
                         VALUES (?, ?, ?, ?, ?)""",
                      (time.time(), from_id, into_id, score, json.dumps(modalities)))

    # ── Manual operations ──
    def merge_entities(self, from_id: str, into_id: str, reason: str = "manual") -> bool:
        if from_id == into_id:
            return False
        with _DB_LOCK, _conn() as c:
            ok = c.execute("SELECT 1 FROM entities WHERE id=?", (from_id,)).fetchone()
            ok2 = c.execute("SELECT 1 FROM entities WHERE id=?", (into_id,)).fetchone()
            if not (ok and ok2):
                return False
            c.execute("UPDATE sightings SET entity_id=? WHERE entity_id=?", (into_id, from_id))
            c.execute("UPDATE entity_aliases SET entity_id=? WHERE entity_id=?", (into_id, from_id))
            c.execute("""INSERT INTO entity_aliases (alias_id, entity_id, merged_ts, score, evidence)
                         VALUES (?, ?, ?, ?, ?)""",
                      (from_id, into_id, time.time(), 1.0, reason))
            c.execute("UPDATE entities SET merged_from=COALESCE(merged_from || ',', '') || ? WHERE id=?",
                      (from_id, into_id))
            c.execute("DELETE FROM entities WHERE id=?", (from_id,))
        return True

    def unmerge_entity(self, alias_id: str) -> bool:
        """Reverse a merge by promoting the alias back to its own entity."""
        with _DB_LOCK, _conn() as c:
            row = c.execute("SELECT entity_id, merged_ts FROM entity_aliases WHERE alias_id=?",
                            (alias_id,)).fetchone()
            if not row:
                return False
            parent, _ = row
            ts = time.time()
            c.execute("""INSERT OR IGNORE INTO entities
                (id, label, created_ts, updated_ts, confidence, sighting_count)
                VALUES (?, ?, ?, ?, ?, 0)""",
                (alias_id, alias_id, ts, ts, 0.5))
            c.execute("DELETE FROM entity_aliases WHERE alias_id=? AND entity_id=?",
                      (alias_id, parent))
        return True

    # ── Queries ──
    def all_entities(self, limit: int = 50) -> list[dict]:
        with _DB_LOCK, _conn() as c:
            rows = c.execute("""SELECT id, label, sighting_count, confidence,
                                       updated_ts, merged_from
                                FROM entities ORDER BY sighting_count DESC LIMIT ?""",
                             (limit,)).fetchall()
        return [{"id": r[0], "label": r[1], "sightings": r[2],
                 "confidence": r[3], "last_seen": r[4],
                 "merged_from": r[5]} for r in rows]

    def entity_detail(self, eid: str) -> dict | None:
        with _DB_LOCK, _conn() as c:
            row = c.execute("""SELECT id, label, created_ts, updated_ts, confidence,
                                      sighting_count, merged_from FROM entities WHERE id=?""",
                            (eid,)).fetchone()
            if not row:
                return None
            sights = c.execute("""SELECT camera, ts, fused_score, modalities
                                  FROM sightings WHERE entity_id=? ORDER BY ts DESC LIMIT 50""",
                               (eid,)).fetchall()
            aliases = c.execute("""SELECT alias_id, score, evidence
                                   FROM entity_aliases WHERE entity_id=?""", (eid,)).fetchall()
        return {
            "id": row[0], "label": row[1],
            "created_ts": row[2], "updated_ts": row[3],
            "confidence": row[4], "sighting_count": row[5],
            "merged_from": row[6],
            "sightings": [{"camera": s[0], "ts": s[1],
                           "score": s[2], "modalities": json.loads(s[3] or "[]")} for s in sights],
            "aliases": [{"alias_id": a[0], "score": a[1], "evidence": a[2]} for a in aliases],
        }

    def merge_log(self, limit: int = 50, only_unreviewed: bool = False) -> list[dict]:
        sql = """SELECT id, ts, from_id, into_id, score, modalities, reviewed
                 FROM merge_log {where} ORDER BY ts DESC LIMIT ?"""
        sql = sql.format(where="WHERE reviewed=0" if only_unreviewed else "")
        with _DB_LOCK, _conn() as c:
            rows = c.execute(sql, (limit,)).fetchall()
        return [{"id": r[0], "ts": r[1], "from": r[2], "into": r[3],
                 "score": r[4], "modalities": json.loads(r[5] or "[]"),
                 "reviewed": bool(r[6])} for r in rows]

    def stats(self) -> dict:
        with _DB_LOCK, _conn() as c:
            ent_n   = c.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            sight_n = c.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
            alias_n = c.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0]
            sugg_n  = c.execute("SELECT COUNT(*) FROM merge_log WHERE reviewed=0").fetchone()[0]
            avg_c   = c.execute("SELECT AVG(confidence) FROM entities").fetchone()[0] or 0.0
        return {"entities": ent_n, "sightings": sight_n, "aliases": alias_n,
                "pending_merge_suggestions": sugg_n, "avg_confidence": round(avg_c, 3)}


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: EntityResolver | None = None


def get_resolver() -> EntityResolver:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = EntityResolver()
    return _INSTANCE


def briefing() -> str:
    r = get_resolver()
    s = r.stats()
    lines = [
        "▸ ENTITY RESOLUTION — COMMAND CENTER",
        f"▸ Resolved entities:        {s['entities']}",
        f"▸ Total sightings:          {s['sightings']}",
        f"▸ Profile aliases linked:   {s['aliases']}",
        f"▸ Avg identity confidence:  {s['avg_confidence']:.0%}",
        f"▸ Pending merge suggestions: {s['pending_merge_suggestions']} (review at /intel/merge_log)",
        f"▸ Fusion modalities active: face(40%), gait(30%), appearance(15%), spatial(10%), profile(5%)",
        f"▸ Merge threshold ≥{MERGE_THRESHOLD:.0%} | Suggest threshold ≥{SUGGEST_THRESHOLD:.0%}",
    ]
    top = r.all_entities(limit=5)
    if top:
        lines.append("▸ TOP ENTITIES BY SIGHTING COUNT:")
        for e in top:
            lines.append(f"   • {e['id']}  '{e['label']}'  sightings={e['sightings']}  conf={e['confidence']:.0%}")
    return "\n".join(lines)
