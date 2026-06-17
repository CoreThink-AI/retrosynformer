#!/usr/bin/env bash
# Deploy the RetroSynFormer dashboard to taco via Tailscale.
# Run from repo root on your local machine.
set -euo pipefail

TACO=taco
REMOTE=/home/hobs/code/corethink/retrosynformer
SERVICE=retrosynformer-dashboard

# Push current branch so taco can pull it.
echo "==> Pushing $(git branch --show-current) to origin"
git push

echo "==> Deploying to $TACO"
ssh "$TACO" bash <<REMOTE
set -euo pipefail
cd $REMOTE

echo "--- git pull"
git pull

echo "--- uv sync --extra dashboard"
~/.local/bin/uv sync --extra dashboard

if [ ! -f .env.dashboard ]; then
  echo ""
  echo "WARNING: .env.dashboard not found."
  echo "  Copy .env.dashboard.example to .env.dashboard and set credentials before"
  echo "  the service will enforce login."
  echo ""
fi

echo "--- install systemd service"
sudo cp deploy/taco-dashboard.service /etc/systemd/system/$SERVICE.service
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE
sudo systemctl restart $SERVICE
sleep 2
sudo systemctl status $SERVICE --no-pager -l
REMOTE

echo ""
echo "Dashboard: http://100.110.98.72:5050/"
echo "           (accessible from any Tailscale-connected device)"
