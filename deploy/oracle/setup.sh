#!/usr/bin/env bash
# Set up ResolveIQ on an Oracle Cloud Always Free server (Ubuntu 24.04, Ampere A1).
#
# Run on the server as the default "ubuntu" user, which has sudo:
#
#   PUBLIC_IP=203.0.113.10 bash setup.sh
#
# or, with your own domain pointing at the server:
#
#   DOMAIN=support.example.com bash setup.sh
#
# Without a domain the site is served at https://<ip-with-dashes>.sslip.io, a free
# public DNS name that resolves to the IP, so Caddy can get a free HTTPS certificate.
# The script can be run again safely; it updates instead of duplicating.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ManasSaxena14/ResolveIQ.git}"
APP_DIR="${APP_DIR:-/opt/resolveiq}"
APP_USER="resolveiq"
ENV_FILE="/etc/resolveiq/resolveiq.env"

say() { printf '\n==> %s\n' "$*"; }

if [[ -z "${DOMAIN:-}" ]]; then
  if [[ -z "${PUBLIC_IP:-}" ]]; then
    echo "Set PUBLIC_IP to the server's public IP (shown in the Oracle console) or DOMAIN." >&2
    exit 1
  fi
  DOMAIN="${PUBLIC_IP//./-}.sslip.io"
fi
if ! grep -q 'VERSION_ID="24.04"' /etc/os-release; then
  echo "This script expects Ubuntu 24.04; other versions may need changes." >&2
fi

say "Installing system packages"
sudo apt-get update -q
sudo apt-get install -y -q python3.12 python3.12-venv python3.12-dev build-essential git \
  curl debian-keyring debian-archive-keyring apt-transport-https iptables-persistent

say "Adding 2 GB of swap for package installs"
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

say "Creating the $APP_USER user and $APP_DIR"
id "$APP_USER" >/dev/null 2>&1 || sudo useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
sudo mkdir -p "$APP_DIR"
sudo chown "$APP_USER:$APP_USER" "$APP_DIR"
as_app() { sudo -u "$APP_USER" -H env HF_HOME="$APP_DIR/.cache/huggingface" "$@"; }

say "Fetching the code"
if [[ -d "$APP_DIR/.git" ]]; then
  as_app git -C "$APP_DIR" pull --ff-only
else
  as_app git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi

say "Installing Python packages (a few minutes the first time)"
[[ -x "$APP_DIR/.venv/bin/python" ]] || as_app python3.12 -m venv "$APP_DIR/.venv"
as_app "$APP_DIR/.venv/bin/pip" install -q --upgrade pip
as_app "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
as_app "$APP_DIR/.venv/bin/pip" install -q -e "$APP_DIR" --no-deps

say "Writing $ENV_FILE (kept if it exists)"
sudo mkdir -p /etc/resolveiq
if [[ ! -f "$ENV_FILE" ]]; then
  sudo tee "$ENV_FILE" >/dev/null <<ENV
# ResolveIQ settings. Restart after editing: sudo systemctl restart resolveiq
RESOLVEIQ_LLM_PROVIDER=groq
# Free tier key from console.groq.com, created on an account without billing.
GROQ_API_KEY=
# Caddy on this machine sets X-Forwarded-For, so each visitor gets their own limit.
RESOLVEIQ_TRUST_PROXY=1
# Optional: comma-separated API keys for programmatic clients.
RESOLVEIQ_API_KEYS=
ENV
fi
sudo chown root:"$APP_USER" "$ENV_FILE"
sudo chmod 640 "$ENV_FILE"

say "Building the knowledge base index"
if [[ ! -f "$APP_DIR/.chroma/index_manifest.json" ]]; then
  (cd "$APP_DIR" && as_app "$APP_DIR/.venv/bin/python" scripts/build_index.py --no-mlflow)
fi

say "Installing the service"
sed "s|@APP_DIR@|$APP_DIR|g" "$APP_DIR/deploy/oracle/resolveiq.service" |
  sudo tee /etc/systemd/system/resolveiq.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable resolveiq >/dev/null

say "Opening ports 80 and 443 in the server firewall"
# Oracle's Ubuntu images reject everything but SSH; allow web traffic before that rule.
for port in 80 443; do
  sudo iptables -C INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT 2>/dev/null ||
    sudo iptables -I INPUT 1 -p tcp -m state --state NEW --dport "$port" -j ACCEPT
done
sudo netfilter-persistent save >/dev/null

say "Installing Caddy for HTTPS"
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' |
    sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' |
    sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -q
  sudo apt-get install -y -q caddy
fi
sed "s|@DOMAIN@|$DOMAIN|g" "$APP_DIR/deploy/oracle/Caddyfile" | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl reload caddy || sudo systemctl restart caddy

if [[ -f "$APP_DIR/models/semantic/semantic_meta.json" &&
      -f "$APP_DIR/models/risk/risk_meta.json" ]]; then
  say "Starting ResolveIQ"
  sudo systemctl restart resolveiq
  echo "Open https://$DOMAIN in a minute, once the models have loaded."
else
  say "Almost done: the trained models are not on the server yet"
  echo "From your computer, in the project folder, run:"
  echo "  make oracle-push HOST=ubuntu@${PUBLIC_IP:-$DOMAIN}"
  echo "It copies the models and the built web app and starts the service."
fi
echo "Add your Groq key in $ENV_FILE, then: sudo systemctl restart resolveiq"
