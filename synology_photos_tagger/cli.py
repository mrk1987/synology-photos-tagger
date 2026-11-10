"""spt - synology-photos-tagger command line.

  spt serve                 run the vision service (GPU)
  spt check                 test settings: database, photo folder, vision service
  spt people-album on|off   switch Synology's People album flags (global + per user)
  spt first-run             scan every photo, group faces into people, write everything
  spt nightly               scan new photos, attach faces to known people, write
  spt undo                  remove everything this tool wrote; unhide Synology's people
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import config, pipeline, store, syno


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="spt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    sub.add_parser("check")
    pa = sub.add_parser("people-album")
    pa.add_argument("state", choices=["on", "off"])
    fr = sub.add_parser("first-run")
    fr.add_argument("--workers", type=int, default=4)
    fr.add_argument("--limit", type=int, default=0, help="analyze at most N photos (testing)")
    fr.add_argument("--no-write", action="store_true", help="scan and group only; inspect before writing")
    nt = sub.add_parser("nightly")
    nt.add_argument("--workers", type=int, default=2)
    sub.add_parser("undo")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        from .vision_service import serve
        serve()
        return

    cfg = config.load()
    pg = syno.connect(cfg)
    uid = syno.user_id(pg, cfg.syno_user)

    if a.cmd == "check":
        import httpx
        units = syno.photo_units(pg, uid)
        print(f"database ok: user '{cfg.syno_user}' (id {uid}) has {len(units)} photos")
        found = sum(1 for _, f, d in units[:50] if pipeline.image_path(cfg, d, f))
        print(f"photo folder {cfg.photos_root}: {found}/{min(50, len(units))} of the first photos found")
        print("vision service:", httpx.get(f"{cfg.vision_url}/health", timeout=10).json())
    elif a.cmd == "people-album":
        syno.people_flags(pg, uid, a.state == "on")
        print(f"People album flags {a.state}")
    elif a.cmd == "undo":
        syno.undo(pg, uid)
        db = store.connect(cfg.data_dir)
        with db:
            db.execute("UPDATE photos SET written = 0")
            db.execute("UPDATE faces SET written = 0")
        print("undone")
    else:
        db = store.connect(cfg.data_dir)
        if a.cmd == "first-run":
            pipeline.scan(cfg, db, pg, uid, workers=a.workers, limit=a.limit)
            if syno.people_written(pg):
                # Regrouping renumbers the groups, and the people already in Synology are tied
                # to the old numbers: faces would land on the wrong person.
                pipeline.log("people are already written: not regrouping, finishing the write. "
                             "To regroup from scratch run `spt undo` first; new photos are handled by `spt nightly`.")
            else:
                pipeline.cluster(cfg, db, pg, uid)
            if a.no_write:
                print("not written (--no-write). Run `spt first-run` again without it to write.")
                return
            pipeline.write(cfg, db, pg, uid)
        else:
            if not db.execute("SELECT 1 FROM faces WHERE cluster > 0 LIMIT 1").fetchone():
                sys.exit("no people yet: run `spt first-run` first")
            done = pipeline.scan(cfg, db, pg, uid, workers=a.workers)
            n = pipeline.assign(cfg, db, done)
            pipeline.log(f"faces attached to known people: {n}")
            pipeline.log(f"new people: {pipeline.new_people(cfg, db)}")
            pipeline.write(cfg, db, pg, uid)
            # For monitoring: age of this file = time since the last successful run.
            # SPT_STATUS_DIR: put it somewhere your monitoring can read (the data dir is mode 700).
            status = Path(os.environ.get("SPT_STATUS_DIR") or cfg.data_dir / "status")
            try:
                status.mkdir(parents=True, exist_ok=True)
                (status / "last_success.json").write_text(json.dumps({"time": int(time.time()), "photos": len(done)}))
            except OSError as exc:
                pipeline.log(f"could not write status file in {status}: {exc}")


if __name__ == "__main__":
    main()
