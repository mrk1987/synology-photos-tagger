# How it works: Synology Photos internals

Found by reading the `synofoto` schema and testing on DSM 7.4.1 / Synology Photos 1.9.1.
Not documented by Synology; may change with any update.

## Tables used

| Table | Use |
|---|---|
| `unit` | One row per file. `type = 0` still image, `1` video. `id_folder` -> `folder.name` (path under the user's `Photos`). |
| `general_tag`, `many_unit_has_many_general_tag` | Tags. Triggers keep counts and bump `unit.version` so clients refresh. |
| `person` | People. `hidden`, `name`, `cover` (a `face.id`). |
| `face` | `bounding_box` / `landmark` JSON in 0-1 image coordinates, `picture` (large object: JPEG face crop, used for avatars), `id_person`, `is_manual`. |
| `many_unit_has_many_person`, `person_item_count` | Photo <-> person links and album counts. `public.recalculate_person_album_item_count(user)` rebuilds the counts. |
| `person_group`, `cluster`, `face.feature` | Synology's own clustering state (protobuf feature vectors). Not used by this tool. |
| `config`, `user_info.config` | Feature flags (below). |

## The People album flags

The People album is only shown when **both** are true:

- `config` row `enable_person = 'true'` (global)
- `user_info.config->>'enable_person' = true` for the user (Personal Space)

Either one alone hides the album. Toggling it in the app also queues a library re-index;
setting the flags in the database does not. With both on, the NAS face-scans **new uploads**
(`unit.index_stage` gains bit 16, e.g. 79 -> 95) but does not rescan the existing library.

## What the tool writes

- Tags into `general_tag` / `many_unit_has_many_general_tag`.
- One `person` per face group, `face` rows with `is_manual = true` and a face-crop large object
  (`GRANT SELECT ... TO "SynologyPhotos"` so the app can read it), links in
  `many_unit_has_many_person`, then `recalculate_person_album_item_count`.
- Synology's own people are set `hidden = true` (previous value kept for undo).
- Every created row id is recorded in schema `tagger`; `spt undo` deletes exactly those.
