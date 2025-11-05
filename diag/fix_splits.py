from pathlib import Path
import re
import pandas as pd

# EDIT THESE to match your case
bids_root = Path("/Users/gm33/data/fairy")
subject   = "003"        # must match filename stem you want
session   = "01"
task      = "test"
run       = None         # or "01" if you really want a run
meg_dir   = bids_root / f"sub-{subject}" / f"ses-{session}" / "meg"

# Build desired stem (no split yet)
parts = [f"sub-{subject}", f"ses-{session}", f"task-{task}"]
if run is not None:
    parts.append(f"run-{run}")
stem = "_".join(parts)

# Find all candidate FIF files that look like splits
pat = re.compile(rf"{re.escape(stem)}(?:_split-(\d+))?_meg\.fif$")
cands = sorted(p for p in meg_dir.glob("*_meg.fif"))

# Map existing candidates to a normalized, contiguous split sequence starting at 1
existing = []
for p in cands:
    m = re.match(r"(.*)_split-(\d+)_meg\.fif$", p.name)
    if m and stem in m.group(1):
        existing.append((p, int(m.group(2))))
    else:
        # also allow a single-file (no split) case accidentally renamed
        if p.name.startswith(stem) and p.name.endswith("_meg.fif"):
            existing.append((p, 1))

if not existing:
    raise SystemExit(f"No matching MEG FIF files found under {meg_dir}")

# Sort by the discovered split number, then rename contiguously to split-01, -02, ...
existing.sort(key=lambda t: t[1])
for idx, (oldp, _old_n) in enumerate(existing, start=1):
    newp = meg_dir / f"{stem}_split-{idx:02d}_meg.fif"
    if oldp == newp:
        continue
    if newp.exists():
        raise FileExistsError(f"Refusing to overwrite {newp}")
    print(f"REN: {oldp.name} -> {newp.name}")
    oldp.rename(newp)

# Ensure sidecars match the stem (no split in their names)
sidecars = [
    ("json", f"{stem}_meg.json"),
    ("tsv",  f"{stem}_channels.tsv"),
    ("json", f"{stem}_coordsystem.json"),  # optional but common
]
for ext, target in sidecars:
    # find any sidecar with _meg.<ext> in this directory and normalize name
    for p in meg_dir.glob(f"*_*_meg.{ext}"):
        # Only rename sidecars that correspond to this recording (heuristic: start with stem without split)
        if p.name.startswith(stem) and "_split-" in p.stem:
            # old bad split-bearing sidecars -> collapse to unsplit name
            newp = meg_dir / target
            if p != newp:
                if newp.exists():
                    print(f"SKIP (exists): {newp.name}")
                else:
                    print(f"REN sidecar: {p.name} -> {newp.name}")
                    p.rename(newp)

# Fix scans.tsv to point to the first split (common practice)
scans_tsv = bids_root / f"sub-{subject}" / f"ses-{session}" / f"sub-{subject}_ses-{session}_scans.tsv"
first_rel = f"meg/{stem}_split-01_meg.fif"

if scans_tsv.exists():
    df = pd.read_csv(scans_tsv, sep="\t")
    # Heuristic: replace any row that referenced old meg file with the new first split
    if "filename" in df.columns:
        df["filename"] = df["filename"].replace(regex=True, to_replace=r"meg/.*_meg\.fif$", value=first_rel)
        df.to_csv(scans_tsv, sep="\t", index=False)
        print(f"Updated scans.tsv -> {scans_tsv}")
else:
    # Create a minimal scans.tsv if missing
    df = pd.DataFrame([{"filename": first_rel}])
    scans_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(scans_tsv, sep="\t", index=False)
    print(f"Created scans.tsv -> {scans_tsv}")
