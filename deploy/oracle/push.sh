#!/usr/bin/env bash
# Copy the trained models and the built web app from this computer to the server,
# then restart the service. Run from the project folder:
#
#   bash deploy/oracle/push.sh ubuntu@203.0.113.10
#
# The server then needs neither Node.js nor a Hugging Face account. Sync-tool
# duplicates such as "model 2.safetensors" and hidden files are never copied.
set -euo pipefail
HOST="${1:?Usage: push.sh ubuntu@SERVER_IP}"
APP_DIR="${APP_DIR:-/opt/resolveiq}"

for path in models/semantic/semantic_meta.json models/risk/risk_meta.json \
            frontend/dist/index.html; do
  [[ -f "$path" ]] || { echo "Missing $path. Train the models and run make web-build." >&2; exit 1; }
done

RSYNC=(rsync -azR --delete --exclude '* [0-9]*' --exclude '.*'
       --rsync-path='sudo rsync')
"${RSYNC[@]}" models/semantic/ models/risk/ frontend/dist/ "$HOST:$APP_DIR/"
ssh "$HOST" "sudo chown -R resolveiq:resolveiq $APP_DIR/models $APP_DIR/frontend/dist && sudo systemctl restart resolveiq && echo 'Service restarted'"
