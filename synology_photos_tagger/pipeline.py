"""scan -> cluster -> write. Used by the CLI for both the first full run and the nightly run."""
from __future__ import annotations

import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

from . import syno
from .config import Config

THUMB = "SYNOPHOTO_THUMB_XL.jpg"
STOP = set((Path(__file__).with_name("stop_tags.txt")).read_text(encoding="utf-8").split("\n")) - {""}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def image_path(cfg: Config, folder: str, filename: str) -> Path | None:
    """Synology's XL thumbnail if present (fast, already a JPEG), else the original file."""
    base = cfg.photos_root / folder.strip("/")
    thumb = base / "@eaDir" / filename / THUMB
    if thumb.is_file():
        return thumb
    orig = base / filename
    return orig if orig.is_file() else None


# ---------------------------------------------------------------- scan
def scan(cfg: Config, db, pg, uid: int, workers: int = 4, limit: int = 0) -> list[int]:
    """Register new photo units and analyze every photo not analyzed yet. Returns analyzed ids."""
    known = {r[0] for r in db.execute("SELECT id FROM photos")}
    new = [(i, f, d) for i, f, d in syno.photo_units(pg, uid) if i not in known]
    db.executemany("INSERT INTO photos(id, filename, folder) VALUES(?,?,?)", new)
    db.commit()
    todo = db.execute("SELECT id, filename, folder FROM photos WHERE analyzed = 0 ORDER BY id").fetchall()
    if limit:
        todo = todo[:limit]
    log(f"photos: {len(new)} new, {len(todo)} to analyze")
    hdr = {"Authorization": f"Bearer {cfg.vision_token}", "Content-Type": "application/octet-stream"}

    def work(row):
        pid, fname, folder = row
        p = image_path(cfg, folder, fname)
        if p is None:
            return pid, None, "missing"
        for delay in (0, 2, 5, 10):
            time.sleep(delay)
            try:
                r = http.post(f"{cfg.vision_url}/analyze?max_side=1600&top_k=60", content=p.read_bytes(),
                              headers=hdr, timeout=180)
                if r.status_code == 200:
                    return pid, r.json(), "ok"
                if r.status_code == 422:
                    return pid, None, "undecodable"
            except (httpx.HTTPError, OSError):
                pass
        return pid, None, "failed"

    done, stats, t0 = [], Counter(), time.time()
    with httpx.Client() as http, ThreadPoolExecutor(workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(work, r) for r in todo]), 1):
            pid, data, status = fut.result()
            stats[status] += 1
            if status == "ok":
                db.execute("DELETE FROM tags WHERE id_unit=?", (pid,))
                db.execute("DELETE FROM faces WHERE id_unit=?", (pid,))
                db.executemany("INSERT INTO tags VALUES(?,?,?)", [(pid, t["tag"], t["score"]) for t in data["tags"]])
                for f in data["faces"]:
                    b = f["bbox_norm"]
                    db.execute("INSERT INTO faces(id_unit,x1,y1,x2,y2,landmarks,score,embedding) VALUES(?,?,?,?,?,?,?,?)",
                               (pid, b["x1"], b["y1"], b["x2"], b["y2"], json.dumps(f["landmarks"]), f["score"],
                                np.asarray(f["embedding"], dtype="<f4").tobytes()))
                db.execute("UPDATE photos SET analyzed=1 WHERE id=?", (pid,))
                done.append(pid)
            elif status == "undecodable":
                db.execute("UPDATE photos SET analyzed=-1 WHERE id=?", (pid,))
            # "missing"/"failed": left at 0, retried next run (thumbnails can lag behind uploads)
            if n % 200 == 0:
                db.commit()
                log(f"  {n}/{len(todo)} {n / (time.time() - t0):.1f}/s {dict(stats)}")
    db.commit()
    log(f"analyzed {dict(stats)}")
    return done


# ---------------------------------------------------------------- cluster (first run)
def _eligible(cfg: Config, score: float, width: float) -> bool:
    return score >= cfg.min_face_score and width >= cfg.min_face_width


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cluster(cfg: Config, db, pg, uid: int, min_size: int = 3) -> None:
    """Group all faces into people (average linkage, cosine distance cut) and carry over the
    names the user already gave in Synology Photos (by box overlap on the same photo)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    rows = db.execute("SELECT id, id_unit, x1, y1, x2, y2, score, embedding FROM faces").fetchall()
    elig = [r for r in rows if _eligible(cfg, r[6], r[4] - r[2])]
    E = np.stack([np.frombuffer(r[7], dtype="<f4") for r in elig]).astype(np.float64)
    raw = fcluster(linkage(E, method="average", metric="cosine"), t=cfg.cluster_dist, criterion="distance")
    size = Counter(raw.tolist())
    order = {c: i for i, (c, _) in enumerate(sorted(((c, n) for c, n in size.items() if n >= min_size),
                                                    key=lambda x: -x[1]), start=1)}
    label = {r[0]: -2 for r in rows}
    for r, c in zip(elig, raw.tolist()):
        label[r[0]] = order.get(c, -1)
    with db:
        db.executemany("UPDATE faces SET cluster=? WHERE id=?", [(c, f) for f, c in label.items()])

    box = {r[0]: (r[2], r[3], r[4], r[5]) for r in rows}
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r[1]].append(r[0])
    votes: dict[int, Counter] = defaultdict(Counter)
    for unit, name, bb in syno.named_faces(pg, uid):
        bb = bb if isinstance(bb, dict) else json.loads(bb)
        sb = (bb["top_left"]["x"], bb["top_left"]["y"], bb["bottom_right"]["x"], bb["bottom_right"]["y"])
        cands = [(_iou(sb, box[f]), f) for f in by_unit.get(unit, [])]
        if cands:
            best, f = max(cands)
            if best >= 0.4 and label[f] > 0:
                votes[label[f]][name] += 1
    with db:
        db.execute("DELETE FROM cluster_names")
        db.executemany("INSERT INTO cluster_names VALUES(?,?,?,?)",
                       [(c, v.most_common(1)[0][0], v.most_common(1)[0][1], sum(v.values())) for c, v in votes.items()])
    split = Counter(v.most_common(1)[0][0] for v in votes.values())
    mixed = sum(1 for v in votes.values() if len(v) > 1)
    log(f"faces {len(rows)}, eligible {len(elig)}, people {len(order)}, named {len(votes)}; "
        f"names split over several people: {sum(n - 1 for n in split.values())}, people with mixed names: {mixed}")


# ---------------------------------------------------------------- assign (nightly)
def assign(cfg: Config, db, unit_ids: list[int]) -> int:
    """Attach faces on the given photos to the nearest known person (no regrouping)."""
    rows = db.execute("SELECT cluster, embedding FROM faces WHERE cluster > 0").fetchall()
    if not rows or not unit_ids:
        return 0
    cl = np.array([r[0] for r in rows])
    E = np.stack([np.frombuffer(r[1], dtype="<f4") for r in rows])
    ids = sorted(set(cl.tolist()))
    cents = np.stack([E[cl == c].mean(axis=0) for c in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    marks = ",".join("?" * len(unit_ids))
    n = 0
    for fid, score, width, emb in db.execute(
            f"SELECT id, score, x2 - x1, embedding FROM faces WHERE id_unit IN ({marks}) AND cluster IS NULL", unit_ids).fetchall():
        c = -2
        if _eligible(cfg, score, width):
            sims = cents @ np.frombuffer(emb, dtype="<f4")
            j = int(sims.argmax())
            c = ids[j] if sims[j] >= cfg.attach else -1
            n += c > 0
        db.execute("UPDATE faces SET cluster=? WHERE id=?", (c, fid))
    db.commit()
    return n


def new_people(cfg: Config, db, min_photos: int = 3) -> int:
    """Group faces that matched nobody (cluster -1, whole library). A group seen on at least
    `min_photos` different photos becomes a new, unnamed person. Returns how many."""
    from scipy.cluster.hierarchy import fcluster, linkage

    pool = db.execute("SELECT id, id_unit, embedding FROM faces WHERE cluster = -1").fetchall()
    if len(pool) < min_photos:
        return 0
    P = np.stack([np.frombuffer(r[2], dtype="<f4") for r in pool]).astype(np.float64)
    lab = fcluster(linkage(P, method="average", metric="cosine"), t=cfg.cluster_dist, criterion="distance")
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (fid, unit, _), g in zip(pool, lab.tolist()):
        groups[g].append((fid, unit))
    next_id = (db.execute("SELECT max(cluster) FROM faces").fetchone()[0] or 0) + 1
    made = 0
    for members in groups.values():
        if len({u for _, u in members}) >= min_photos:
            db.executemany("UPDATE faces SET cluster=? WHERE id=?", [(next_id, f) for f, _ in members])
            next_id += 1
            made += 1
    db.commit()
    return made


# ---------------------------------------------------------------- write
def _crop(cfg: Config, folder: str, fname: str, box) -> bytes | None:
    p = image_path(cfg, folder, fname)
    if p is None:
        return None
    try:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    except OSError:
        return None
    w, h = im.size
    cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
    s = max((box[2] - box[0]) * w, (box[3] - box[1]) * h) * 1.4
    c = im.crop((max(0, int(cx - s / 2)), max(0, int(cy - s / 2)), min(w, int(cx + s / 2)), min(h, int(cy + s / 2))))
    if min(c.size) < 8:
        return None
    c.thumbnail((192, 192))
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def write(cfg: Config, db, pg, uid: int, batch: int = 500) -> None:
    """Write everything analyzed but not yet written: tags, new people (first run), faces.
    Commits in batches; safe to interrupt and re-run. Recorded in the `tagger` ledger."""
    pg.execute(syno.LEDGER)
    pg.commit()

    # Tags
    units = [r[0] for r in db.execute("SELECT id FROM photos WHERE analyzed = 1 AND written = 0 ORDER BY id")]
    log(f"writing tags for {len(units)} photos")
    for i in range(0, len(units), batch):
        chunk = units[i:i + batch]
        pairs = []
        for u in chunk:
            ts = [t.lower() for (t,) in db.execute("SELECT tag FROM tags WHERE id_unit=? ORDER BY score DESC", (u,))]
            pairs += [(u, t) for t in [t for t in ts if t not in STOP][:cfg.max_tags]]
        with pg.transaction():
            pg.execute("CREATE TEMP TABLE IF NOT EXISTS pt(id_unit int, name text) ON COMMIT DELETE ROWS")
            with pg.cursor().copy("COPY pt FROM STDIN") as cp:
                for row in pairs:
                    cp.write_row(row)
            pg.execute("""WITH ins AS (
                INSERT INTO general_tag(id_user, name, normalized_name, name_for_sort)
                SELECT DISTINCT %(u)s, name, name, name FROM pt
                WHERE NOT EXISTS (SELECT 1 FROM general_tag g WHERE g.id_user = %(u)s AND g.name = pt.name)
                RETURNING id) INSERT INTO tagger.created_tag SELECT id FROM ins""", {"u": uid})
            pg.execute("""WITH ins AS (
                INSERT INTO many_unit_has_many_general_tag(id_unit, id_general_tag)
                SELECT DISTINCT pt.id_unit, g.id FROM pt JOIN general_tag g ON g.id_user = %(u)s AND g.name = pt.name
                JOIN unit ON unit.id = pt.id_unit
                WHERE NOT EXISTS (SELECT 1 FROM many_unit_has_many_general_tag m
                                  WHERE m.id_unit = pt.id_unit AND m.id_general_tag = g.id)
                RETURNING id_unit, id_general_tag)
                INSERT INTO tagger.tag_link SELECT * FROM ins ON CONFLICT DO NOTHING""", {"u": uid})
        db.executemany("UPDATE photos SET written=1 WHERE id=?", [(u,) for u in chunk])
        db.commit()
        log(f"  tags {min(i + batch, len(units))}/{len(units)}")

    # People: create a Synology person for every cluster that does not have one yet.
    names = dict(db.execute("SELECT cluster, name FROM cluster_names"))
    have = {c for (c,) in pg.execute("SELECT cluster FROM tagger.person")}
    clusters = [c for (c,) in db.execute(
        "SELECT cluster FROM faces WHERE cluster > 0 GROUP BY cluster HAVING count(DISTINCT id_unit) >= 2")]
    with pg.transaction():
        for c in clusters:
            if c not in have:
                name = names.get(c, "")
                pg.execute("""WITH p AS (INSERT INTO person(id_user, name, normalized_name, hidden, created_time)
                              VALUES (%s, %s, %s, false, extract(epoch from now())::bigint) RETURNING id)
                              INSERT INTO tagger.person SELECT id, %s FROM p""", (uid, name, name.lower(), c))
    # Faces
    faces = db.execute("SELECT f.id, f.id_unit, f.x1, f.y1, f.x2, f.y2, f.score, f.cluster, p.folder, p.filename "
                       "FROM faces f JOIN photos p ON p.id = f.id_unit WHERE f.cluster > 0 AND f.written = 0 "
                       "ORDER BY f.cluster, f.score DESC").fetchall()
    person = dict(pg.execute("SELECT cluster, id FROM tagger.person"))
    log(f"writing {len(faces)} faces")
    for i in range(0, len(faces), batch):
        chunk = [f for f in faces[i:i + batch] if f[7] in person]
        with pg.transaction():
            for fid, unit, x1, y1, x2, y2, score, c, folder, fname in chunk:
                jpg = _crop(cfg, folder, fname, (x1, y1, x2, y2))
                bb = json.dumps({"top_left": {"x": x1, "y": y1}, "bottom_right": {"x": x2, "y": y2}})
                pid = person[c]
                row = pg.execute("""INSERT INTO face(id_user, bounding_box, picture, score, id_unit, ref_id_unit, id_person, is_manual)
                    VALUES (%s, %s::json, CASE WHEN %s::bytea IS NULL THEN NULL ELSE lo_from_bytea(0, %s::bytea) END,
                            %s, %s, %s, %s, true) RETURNING id, picture""",
                                 (uid, bb, jpg, jpg, int(score * 1e7), unit, unit, pid)).fetchone()
                pg.execute("INSERT INTO tagger.face VALUES (%s, %s)", row)
                if row[1] is not None:
                    pg.execute(f'GRANT SELECT ON LARGE OBJECT {int(row[1])} TO "SynologyPhotos"')
                pg.execute("UPDATE person SET cover = %s WHERE id = %s AND cover IS NULL", (row[0], pid))
                pg.execute("""WITH m AS (INSERT INTO many_unit_has_many_person(id_unit, id_person, id_user)
                    SELECT %(unit)s, %(p)s, %(u)s WHERE NOT EXISTS
                      (SELECT 1 FROM many_unit_has_many_person WHERE id_unit = %(unit)s AND id_person = %(p)s)
                    RETURNING id_unit, id_person) INSERT INTO tagger.person_unit SELECT * FROM m ON CONFLICT DO NOTHING""",
                           {"unit": unit, "p": pid, "u": uid})
        db.executemany("UPDATE faces SET written=1 WHERE id=?", [(f[0],) for f in chunk])
        db.commit()
        log(f"  faces {min(i + batch, len(faces))}/{len(faces)}")

    with pg.transaction():
        pg.execute(syno.FINISH, {"u": uid})
    log("done: Synology's own people hidden, person counts recalculated")
