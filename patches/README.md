# patches/ - Rust changes to Brush that the server depends on

`brush/` is a separate upstream repository (git-ignored here). The server trains with a **patched** Brush:

| Item | Value |
|---|---|
| Upstream base | `3edecbb` = release **v0.3.0** of https://github.com/ArthurBrussee/brush |
| Patched branch | `gs-server-patches`, commit `b0900e6`, in `/home/anoma/brush` **on the server only** (origin there is upstream - never push) |
| Files here | `brush_gs-server-patches.patch` (`git format-patch`, 19 files, +565/-47) and `brush_gs-server-patches.bundle` (`3edecbb..gs-server-patches`) |
| Verified | the patch applies cleanly to a clean `3edecbb` snapshot (2026-10-19) |

What the patch adds (all opt-in, off by default; the PLY is byte-identical when every flag is at its default):
a depth render channel + `--depth-loss-weight/--depth-max-range`, `--position-anchor-weight/--position-anchor-max-dist`,
`--aniso-reg-weight/--aniso-reg-ratio-threshold`, and `--photo-consistency-enabled/--photo-consistency-warmup-steps`
(optional PLY properties `photo_resid_avg/photo_resid_count`). Penalty terms are normalised by the number of
*violating* splats - a plain `.sum()` over ~1.7 M splats explodes or vanishes (see `CLAUDE.md`).

## Restore / rebuild from this folder

```bash
git clone https://github.com/ArthurBrussee/brush.git brush-work && cd brush-work
git checkout 3edecbb
git apply ../patches/brush_gs-server-patches.patch        # or: git am ../patches/brush_gs-server-patches.patch
# or from the bundle:   git fetch ../patches/brush_gs-server-patches.bundle gs-server-patches:gs-server-patches
source ~/.cargo/env && cargo build --release --bin brush_app      # ~3.5 min; GS_BRUSH_EXE points at target/release/brush_app
```

Do **not** apply it to the repository's local `brush/` checkout: that one sits on a different upstream commit
(`83b7c71f`) and carries 37 lines of older local edits.

## Refresh this copy after changing Rust code on the server

The server has no git identity: pass one per command and commit on `gs-server-patches`.

```bash
cd /home/anoma/brush
git -c user.name=<name> -c user.email=<email> commit -am "message"
git bundle create /home/anoma/brush_backup/brush_gs-server-patches.bundle 3edecbb..gs-server-patches
git format-patch 3edecbb..gs-server-patches --stdout > /home/anoma/brush_backup/brush_gs-server-patches.patch
```

Then `gcloud compute scp` both files into this folder and commit them.
