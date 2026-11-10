# synology-photos-tagger

Better people and tags for **Synology Photos**, computed on your own GPU instead of the NAS.

- **People**: faces are detected and grouped with InsightFace (ArcFace embeddings), then written
  into Synology Photos' own People album. You name and merge people in the app as usual.
  Names you already gave in Synology are carried over automatically.
- **Tags**: every photo gets up to 15 tags from RAM++ (Recognize Anything, 4,585 tags:
  objects, scenes, events, activities), searchable in Synology Photos and listed under Tags.
- **NAS stays idle**: the heavy work runs on another PC, VM or container. An NVIDIA GPU is
  recommended (a 4 GB GTX 1650 is enough: about 0.4 s per photo) but not required: without one
  everything falls back to the CPU, only slower. See "No GPU?" in [docs/SETUP.md](docs/SETUP.md).
- **Reversible**: every row it writes is recorded in a separate `tagger` schema;
  `spt undo` removes exactly those rows. Synology's own people are hidden, never deleted.

> **Warning.** This writes to Synology Photos' internal PostgreSQL database, which is not a
> public API. Back up the database first (`pg_dump`, see docs/SETUP.md). A DSM or Synology Photos
> update may change the schema or reset the access settings. Use at your own risk.

> **Privacy.** Face embeddings are biometric data. They stay on your machine in `SPT_DATA_DIR`.
> Photos of other people (guests, children) end up in there too.

> **Status: 0.1, early.** The results below come from the scripts this package was distilled
> from. The packaged `spt` commands have not yet been run end to end on a fresh install.
> Try `spt first-run --limit 200 --no-write` first, and please open an issue with what you find.

## Results on a real library

23,600 photos on Synology Photos 1.9.1 / DSM 7.4:

| | Synology's built-in | synology-photos-tagger |
|---|---|---|
| People | 2,477 groups, mostly 1-3 photos each | 348 people, 2,000+ named faces with 0 mix-ups |
| Tags | object recognition off (poor quality) | 2,977 distinct tags, 201k links |
| NAS CPU | constant during indexing | one face scan per upload |
| First run | - | ~50 min on an RTX 5090, a few hours on a GTX 1650 |

## How it works

```
NAS (Synology Photos)                          GPU host (PC, VM or LXC)
  synofoto PostgreSQL  <--- SSH tunnel --->      spt nightly / first-run
  Photos folder        <--- read-only mount -->  spt serve  (InsightFace + RAM++ on ONNX Runtime)
```

1. `spt first-run` reads the photo list from synofoto, sends each Synology thumbnail (or the
   original) to the vision service, groups all faces (average-linkage on cosine distance 0.65),
   matches the groups to the names you already gave in Synology, and writes tags, people and faces.
2. `spt nightly` does the same for new uploads only and attaches their faces to known people.
   Synology still face-scans new uploads itself (the People album needs that feature on);
   any new person it creates is hidden again by the nightly run.

Details of the Synology internals this relies on: [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

## Install

See **[docs/SETUP.md](docs/SETUP.md)**: NAS access (database role, SSH tunnel), models,
the vision service, the first run and the nightly timer.

```
pip install .
cp .env.example .env    # fill in
spt check
spt first-run --limit 200 --no-write   # try on 200 photos, inspect
spt first-run
```

## TODO

- **Video tagging.** Include videos (`unit.type = 1`): sample 5-10 frames per video with
  `ffmpeg` keyframe seeks, run the tag model on each and merge the tags. Estimated 2-4 s per
  video on a GTX 1650 (not measured). Start with tags only. Open questions: whether Synology
  Photos shows and searches tags on videos like on photos (test on one video first), and how
  the app handles a `face` row that points at a video.

## Credits

- Inspired by [eleonne/synology-tagger](https://github.com/eleonne/synology-tagger)
  (Apache-2.0), which first showed that tags written to `general_tag` /
  `many_unit_has_many_general_tag` appear in Synology Photos. This is a rewrite; no code from
  it is used. See [NOTICE](NOTICE).
- Faces: [InsightFace](https://github.com/deepinsight/insightface) (code MIT). **The antelopev2
  model weights are licensed for non-commercial research use only** and are not included here;
  they are downloaded from InsightFace. Check the licence before any commercial use.
- Tags: [RAM++ / Recognize Anything](https://github.com/xinyu1205/recognize-anything)
  (Apache-2.0), ONNX export by CannotFindObject, mirrored at
  [imagent/ram-plus-onnx](https://huggingface.co/imagent/ram-plus-onnx).

Not affiliated with Synology Inc.

## Licence

Apache-2.0, see [LICENSE](LICENSE).
