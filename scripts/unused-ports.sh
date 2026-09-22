#!/usr/bin/env bash
# Repository port methods that no upper layer ever calls.
#
# This found `tool_results.list_by_candidate`: the build and test output was
# written to the database on every validation and read back by nobody, so the
# coder repaired blind and three real runs burned their whole budget deriving
# the same code. The data was there the entire time.
#
# Worth re-running after adding a port method. A method nothing calls is
# either dead weight or, as it was that time, a conclusion nobody collects.
set -Eeuo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

.venv/bin/python - <<'PY'
import re
import subprocess
from pathlib import Path

declared = sorted(
    set(re.findall(r"    async def (\w+)\(", Path("src/domain/ports/repositories.py").read_text()))
)
callers = subprocess.run(
    ["grep", "-rho", "--include=*.py", r"\.\w\+(", "src/application", "src/interfaces"],
    capture_output=True,
    text=True,
).stdout
used = set(re.findall(r"\.(\w+)\(", callers))

orphans = [name for name in declared if name not in used and not name.startswith("_")]
print(f"{len(declared)} port methods, {len(orphans)} never called from application/ or interfaces/")
for name in orphans:
    print(f"  - {name}")
PY
