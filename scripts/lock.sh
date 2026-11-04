#!/usr/bin/env bash
# Regenerate requirements.lock with exact versions AND sha256 hashes (supply-chain pinning).
# The hashes are produced by pip-compile against PyPI when you run this; they are NOT checked in
# by default because they must be generated on a machine with network access. Commit the result.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --quiet "pip-tools>=7.4"
python -m piptools compile --generate-hashes --allow-unsafe --resolver=backtracking \
  --output-file requirements.lock requirements.txt
echo "Wrote requirements.lock. Docker builds will now use: pip install --require-hashes -r requirements.lock"
