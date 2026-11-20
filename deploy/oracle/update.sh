#!/usr/bin/env bash
# Update ResolveIQ on the Oracle Cloud server to the latest code and restart it.
# Run on the server: bash /opt/resolveiq/deploy/oracle/update.sh
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/resolveiq}"
as_app() { sudo -u resolveiq -H env HF_HOME="$APP_DIR/.cache/huggingface" "$@"; }

as_app git -C "$APP_DIR" pull --ff-only
as_app "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
as_app "$APP_DIR/.venv/bin/pip" install -q -e "$APP_DIR" --no-deps
sudo systemctl restart resolveiq
echo "Updated to $(as_app git -C "$APP_DIR" log -1 --format='%h %s')"
