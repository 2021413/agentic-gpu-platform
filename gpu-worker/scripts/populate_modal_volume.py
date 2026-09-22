"""Put the weights on the Modal Volume, once, without a GPU.

    modal run scripts/populate_modal_volume.py
    modal run scripts/populate_modal_volume.py --force     # ignore the marker
    modal run scripts/populate_modal_volume.py --verify    # check, download nothing

Run this before the first deployment, and again whenever `MODEL_ID` or
`MODEL_REVISION` changes.

## Why a separate CPU job

Downloading 31.2 GB takes tens of minutes. Doing it inside the GPU Server means
an H100 sits at $0.001097/second watching a network transfer it contributes
nothing to — roughly two and a half dollars to move bytes a CPU container moves
for a fraction of a cent. Section 30 of the specification states the rule; this
script is what makes the rule keepable, and `infra/modal/runtime.py` is what
refuses to start a GPU when somebody forgets to run it.

Idempotent by construction: `prepare_model` returns immediately when the marker
matches and the snapshot still verifies, so running this twice costs one
directory listing.
"""

from __future__ import annotations

from pathlib import Path

import modal

from infra.modal.image import worker_image
from infra.modal.secrets import worker_secrets
from infra.modal.volumes import model_cache_volume, volume_mounts
from worker.config import WorkerConfig
from worker.filesystem import GIB, disk_report
from worker.model_state import prepare_model, read_marker, verify_snapshot

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(
    "agentic-gpu-populate",
    tags={"service": "gpu-worker", "project": "agentic", "role": "populate"},
)


@app.function(
    image=worker_image,
    volumes=volume_mounts(),
    secrets=worker_secrets(),
    # No `gpu=`. That absence is the point of this file.
    cpu=8.0,
    memory=16384,
    # A cold 31.2 GB download over four shards, with retries, on a bad day.
    timeout=4 * HOURS,
    # One writer. Modal Volumes are last-write-wins and do not implement
    # advisory locking, so the mutual exclusion the RunPod image got from
    # `flock` has to come from the autoscaler here instead.
    max_containers=1,
)
def populate(force: bool = False, verify_only: bool = False) -> dict[str, object]:
    """Download and verify the model on the Volume. Returns what it found or did."""
    config = WorkerConfig.from_env()
    layout = config.layout
    for directory in layout.all_directories():
        directory.mkdir(parents=True, exist_ok=True)

    # Another populate run may have committed since this container started.
    model_cache_volume.reload()

    report = disk_report(layout.root)
    print(f"volume: {report.free_gb:,.1f} GB free of {report.total_gb:,.1f} GB at {layout.root}")
    print(f"model:  {config.model_id} @ {config.model_revision or 'main (unpinned)'}")

    marker = read_marker(layout)
    if marker is not None:
        ok, problems = verify_snapshot(Path(marker.snapshot_path), marker.files)
        print(
            f"marker: {marker.model_id} @ {marker.revision}, "
            f"{marker.total_bytes / GIB:,.1f} GB, {'verified' if ok else 'PROBLEMS'}"
        )
        for problem in problems[:10]:
            print(f"  {problem}")
        if verify_only:
            return {
                "action": "verify",
                "model_id": marker.model_id,
                "revision": marker.revision,
                "verified": ok,
                "problems": problems[:50],
                "total_bytes": marker.total_bytes,
            }
    elif verify_only:
        print("marker: absent — nothing has been prepared on this volume")
        return {"action": "verify", "verified": False, "problems": ["no marker"]}

    if force and marker is not None:
        # The marker goes, the blobs stay. Deleting the cache to "start clean"
        # is how a flaky network becomes an unbounded bill: the hub client
        # resumes partial files, and it can only resume what still exists.
        layout.model_ready_marker.unlink(missing_ok=True)
        print("--force: marker removed, blobs kept so the download can resume")

    outcome = prepare_model(config, log=print)

    # Background commits run every few seconds and one more happens at exit, but
    # an explicit commit here is what makes the next line of output true.
    model_cache_volume.commit()

    print(
        f"{'downloaded' if outcome.downloaded else 'reused'} "
        f"{outcome.state.total_bytes / GIB:,.1f} GB in {outcome.duration_seconds:.0f}s "
        f"({outcome.attempts} attempt(s))"
    )
    return {
        "action": "download" if outcome.downloaded else "reuse",
        "model_id": outcome.state.model_id,
        "revision": outcome.state.revision,
        "snapshot_path": outcome.state.snapshot_path,
        "total_bytes": outcome.state.total_bytes,
        "verified": outcome.state.verified,
        "duration_seconds": outcome.duration_seconds,
        "attempts": outcome.attempts,
    }


@app.local_entrypoint()
def main(force: bool = False, verify: bool = False) -> None:
    result = populate.remote(force=force, verify_only=verify)
    print()
    for key in sorted(result):
        print(f"{key:>16}  {result[key]}")
