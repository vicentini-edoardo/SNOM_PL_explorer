#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x ".venv/bin/python" ]]; then
  exec ".venv/bin/python" app.py "$@"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  exec python app.py "$@"
fi

cat <<'EOF' >&2
Missing Python environment for SNOM_PL_explorer.

From this folder, run:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt

Then launch again with:
  ./launch_macos.sh
EOF
exit 1
