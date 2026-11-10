"""Access to Synology Photos' database (synofoto) and the `tagger` ledger schema.

Every row this tool creates is recorded in schema `tagger`, so `undo()` removes exactly those
rows. Synology's own people are hidden (never deleted) and restored on undo.
"""
from __future__ import annotations

import psycopg

from .config import Config

LEDGER = """
CREATE SCHEMA IF NOT EXISTS tagger;
CREATE TABLE IF NOT EXISTS tagger.tag_link(id_unit int, id_general_tag int, PRIMARY KEY(id_unit, id_general_tag));
CREATE TABLE IF NOT EXISTS tagger.created_tag(id int PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tagger.person(id int PRIMARY KEY, cluster int UNIQUE);
CREATE TABLE IF NOT EXISTS tagger.face(id int PRIMARY KEY, picture oid);
CREATE TABLE IF NOT EXISTS tagger.person_unit(id_unit int, id_person int, PRIMARY KEY(id_unit, id_person));
CREATE TABLE IF NOT EXISTS tagger.hidden_person(id int PRIMARY KEY, prev_hidden boolean);
"""

FINISH = """
INSERT INTO tagger.hidden_person(id, prev_hidden)
  SELECT id, hidden FROM person WHERE id_user = %(u)s AND id NOT IN (SELECT id FROM tagger.person)
  ON CONFLICT (id) DO NOTHING;
UPDATE person SET hidden = true WHERE id IN (SELECT id FROM tagger.hidden_person) AND hidden = false;
SELECT public.recalculate_person_album_item_count(%(u)s);
"""

UNDO = """
DELETE FROM many_unit_has_many_general_tag m USING tagger.tag_link t
  WHERE m.id_unit = t.id_unit AND m.id_general_tag = t.id_general_tag;
DELETE FROM general_tag g USING tagger.created_tag c WHERE g.id = c.id
  AND NOT EXISTS (SELECT 1 FROM many_unit_has_many_general_tag m WHERE m.id_general_tag = g.id);
DELETE FROM many_unit_has_many_person m USING tagger.person_unit p
  WHERE m.id_unit = p.id_unit AND m.id_person = p.id_person;
UPDATE person SET cover = NULL WHERE id IN (SELECT id FROM tagger.person);
DELETE FROM face WHERE id IN (SELECT id FROM tagger.face);
SELECT lo_unlink(f.picture) FROM tagger.face f
  WHERE f.picture IS NOT NULL AND EXISTS (SELECT 1 FROM pg_largeobject_metadata m WHERE m.oid = f.picture);
DELETE FROM person_item_count WHERE id_person IN (SELECT id FROM tagger.person);
DELETE FROM person WHERE id IN (SELECT id FROM tagger.person);
UPDATE person p SET hidden = h.prev_hidden FROM tagger.hidden_person h WHERE p.id = h.id;
"""


def connect(cfg: Config) -> psycopg.Connection:
    # search_path=public: if the role is named like the ledger schema (`tagger`), Postgres'
    # default "$user", public would resolve `face` to tagger.face.
    # autocommit: each `with pg.transaction()` is then a real transaction, not a savepoint
    # inside an implicit one that is rolled back on exit.
    return psycopg.connect(cfg.pg_dsn, password=cfg.pg_password or None, connect_timeout=15,
                           options="-c search_path=public", autocommit=True)


def user_id(pg: psycopg.Connection, name: str) -> int:
    row = pg.execute("SELECT id FROM user_info WHERE name = %s", (name,)).fetchone()
    if not row:
        raise SystemExit(f"Synology Photos user '{name}' not found in user_info")
    return row[0]


def photo_units(pg: psycopg.Connection, uid: int) -> list[tuple[int, str, str]]:
    """(unit id, filename, folder path relative to the user's Photos root) for still images."""
    return pg.execute(
        "SELECT u.id, u.filename, f.name FROM unit u JOIN folder f ON f.id = u.id_folder "
        "WHERE u.type = 0 AND u.id_user = %s ORDER BY u.id", (uid,)).fetchall()


def named_faces(pg: psycopg.Connection, uid: int) -> list[tuple[int, str, dict]]:
    """Synology's own faces that belong to a named person: (unit, name, bounding_box)."""
    sql = ("SELECT f.id_unit, p.name, f.bounding_box FROM face f JOIN person p ON p.id = f.id_person "
           "WHERE f.id_user = %s AND p.name <> ''")
    if ledger_exists(pg):  # ignore faces this tool wrote itself
        sql += " AND f.id NOT IN (SELECT id FROM tagger.face)"
    return pg.execute(sql, (uid,)).fetchall()


def ledger_exists(pg: psycopg.Connection) -> bool:
    return pg.execute("SELECT to_regclass('tagger.face') IS NOT NULL").fetchone()[0]


def people_written(pg: psycopg.Connection) -> bool:
    """True once this tool has created people in Synology Photos."""
    if not pg.execute("SELECT to_regclass('tagger.person') IS NOT NULL").fetchone()[0]:
        return False
    return pg.execute("SELECT EXISTS (SELECT 1 FROM tagger.person)").fetchone()[0]


def people_flags(pg: psycopg.Connection, uid: int, on: bool) -> None:
    """The People album needs both the global and the per-user enable_person flag."""
    v = "true" if on else "false"
    with pg.transaction():
        pg.execute("UPDATE config SET value = %s WHERE key = 'enable_person'", (v,))
        pg.execute("UPDATE user_info SET config = jsonb_set(config::jsonb, '{enable_person}', %s::jsonb)::json "
                   "WHERE id = %s", (v, uid))


def undo(pg: psycopg.Connection, uid: int) -> None:
    if not ledger_exists(pg):
        print("nothing to undo: no tagger schema")
        return
    with pg.transaction():
        pg.execute(UNDO)
        pg.execute("SELECT public.recalculate_person_album_item_count(%s)", (uid,))
        pg.execute("DROP SCHEMA tagger CASCADE")
