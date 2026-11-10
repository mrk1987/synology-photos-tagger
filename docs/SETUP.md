# Setup

Tested on DSM 7.4.1, Synology Photos 1.9.1 (PostgreSQL 11), Debian 13 LXC on Proxmox with a
GTX 1650 Super (4 GB), and Windows 11 with an RTX 5090. Commands marked **NAS** run over SSH on
the NAS as an administrator account (here called `admin-user`).

## 0. Back up the database (NAS)

```
sudo -u postgres pg_dump -Fc synofoto > ~/synofoto-backup-$(date +%F).dump
```

Restore if ever needed (stop Synology Photos first):
`sudo -u postgres pg_restore --clean --if-exists -d synofoto <file>`

## 1. GPU host

Any Linux or Windows machine with an NVIDIA GPU (4 GB+), CUDA driver 580 or newer, Python 3.11+,
and read access to your Photos folder.

**Photos folder.** Mount the NAS share `homes` (or just `homes/<user>/Photos`) read-only, and
point `SPT_PHOTOS_ROOT` at the `Photos` folder. Files under Synology homes are often mode 700
owned by the NAS user: in an unprivileged LXC they are unreadable, so use a privileged
container, an idmap to that uid, or mount the share as that user.

**LXC on Proxmox.** Pass the GPU in the same way as for Plex/Jellyfin (`/dev/nvidia*` device
entries), and install the NVIDIA userspace driver inside the container with exactly the host's
version (`./NVIDIA-Linux-x86_64-<ver>.run --no-kernel-modules`). Re-run it after host driver
upgrades.

```
python3 -m venv /opt/synology-photos-tagger/.venv
. /opt/synology-photos-tagger/.venv/bin/activate
pip install .
# insightface pulls the CPU-only onnxruntime, which hides the GPU build:
pip uninstall -y onnxruntime && pip install --force-reinstall --no-deps "onnxruntime-gpu[cuda,cudnn]"
```

**Models** (into `SPT_MODEL_DIR`):

```
sh scripts/download_models.sh /opt/synology-photos-tagger/models
```

This fetches RAM++ ONNX (Apache-2.0, ~1.8 GB) and InsightFace antelopev2 (**non-commercial
research licence**, ~350 MB).

**No GPU?** The vision service uses ONNX Runtime, which falls back to the CPU when CUDA is not
available, so the same install works without a GPU. Expect seconds per photo instead of 0.4 s:
the tag model alone measured 2-8 s per photo on CPU. That is fine for nightly runs; a first run
over a large library takes days, and can be stopped and resumed. To make faces lighter on CPU,
set `SPT_FACE_MODEL=buffalo_s` (InsightFace's small pack, same non-commercial licence,
downloaded automatically on first start). This is **untested**: it is less accurate than
antelopev2, and `SPT_CLUSTER_DIST` / `SPT_ATTACH` were tuned for antelopev2, so check the
summary of `spt first-run --no-write` before writing. Pick the pack before the first run;
embeddings from different packs cannot be mixed.

## 2. Database access (NAS)

The tool needs a PostgreSQL role on `synofoto`. Postgres on DSM only listens on 127.0.0.1, so
the GPU host reaches it through an SSH tunnel; nothing is opened to the network.

**Role.** Member of Synology Photos' own role (the database triggers touch many tables), not
superuser:

```
sudo -u postgres psql -d synofoto <<'SQL'
CREATE ROLE tagger LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD 'choose-a-long-random-password';
GRANT CONNECT ON DATABASE synofoto TO tagger;
GRANT "SynologyPhotos" TO tagger;
GRANT CREATE ON DATABASE synofoto TO tagger;   -- for the `tagger` ledger schema
SQL
```

**pg_hba.** Allow that role from localhost only, then reload:

```
sudo cp -p /etc/postgresql/pg_hba.conf /etc/postgresql/pg_hba.conf.bak
echo 'host synofoto tagger 127.0.0.1/32 md5' | sudo tee -a /etc/postgresql/pg_hba.conf
sudo -u postgres psql -c 'SELECT pg_reload_conf()'
```

**SSH tunnel.** DSM disables port forwarding for everyone except root/admin, and refuses keys
in a world-writable home. Allow forwarding to Postgres only, for one account:

```
chmod 755 ~                                   # DSM homes are often 777
sudo cp -p /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
printf '\nMatch User admin-user\n\tAllowTcpForwarding local\n\tPermitOpen 127.0.0.1:5432\n' | sudo tee -a /etc/ssh/sshd_config
sudo sshd -t && sudo systemctl restart sshd   # sshd -t validates first
```

On the GPU host, create a key and authorize it on the NAS restricted to that one forward:

```
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
# on the NAS, in ~admin-user/.ssh/authorized_keys (mode 600, .ssh mode 700):
restrict,port-forwarding,permitopen="127.0.0.1:5432",command="/bin/false" ssh-ed25519 AAAA... gpu-host
```

Run the tunnel as a service (example unit in `deploy/`):
`ssh -N -L 127.0.0.1:15432:127.0.0.1:5432 admin-user@nas`

A DSM update can reset `sshd_config` and `pg_hba.conf`. If the tool suddenly cannot connect,
check both.

## 3. Configure and test

```
cp .env.example .env     # fill in; generate SPT_VISION_TOKEN with secrets.token_hex(32)
spt serve &              # or the systemd unit
spt check
```

## 4. People album

Synology only shows the People album when face recognition is enabled both globally and for
the user. Turning it on in the app also queues a full re-index on the NAS; setting the flags
directly does not:

```
spt people-album on
```

The NAS will still face-scan new uploads (light). The nightly run hides the people it creates.

## 5. First run

```
spt first-run --limit 200 --no-write    # test: scan 200 photos and group, write nothing
spt first-run --no-write                 # everything, then read the summary line
spt first-run                            # writes tags, people, faces
```

The summary prints how many of your existing Synology names ended up split over several people
or mixed into one; on the tested library both were 0 at `SPT_CLUSTER_DIST=0.65`.

## 6. Nightly

Install the units from `deploy/` (vision service, tunnel, nightly timer):

```
cp deploy/*.service deploy/*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now spt-vision.service spt-tunnel.service spt-nightly.timer
```

## Undo

```
spt undo                 # removes every row the tool wrote, unhides Synology's people
spt people-album off     # optional
```
