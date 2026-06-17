#!/usr/bin/env bash
# Deploy the RetroSynFormer dashboard to taco via Tailscale.
# Run from repo root on your local machine.
#
# First-time only: stop biochem-admin if it occupies port 5050
#   ssh taco
#   sudo systemctl stop biochem-admin.service
#   sudo systemctl disable biochem-admin.service
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
git pull --ff-only || git merge --no-ff origin/\$(git rev-parse --abbrev-ref HEAD)

echo "--- install/update dashboard deps in .venv-rocm"
~/.local/bin/uv pip install --python .venv-rocm/bin/python \
  "flask>=3.0,<4" "flask-admin>=2.0,<3" "flask-sqlalchemy>=3.1,<4" "wtforms>=3.1,<4"

echo "--- reinstall retrosynformer (picks up new scripts)"
~/.local/bin/uv pip install --python .venv-rocm/bin/python -e . --no-deps

if [ ! -f .env.dashboard ]; then
  echo ""
  echo "WARNING: .env.dashboard not found."
  echo "  Copy .env.dashboard.example to .env.dashboard and fill in credentials."
  echo ""
fi

echo "--- install user systemd service"
mkdir -p ~/.config/systemd/user
cp deploy/taco-dashboard.user.service ~/.config/systemd/user/$SERVICE.service
systemctl --user daemon-reload
systemctl --user enable $SERVICE
loginctl enable-linger hobs 2>/dev/null || true
systemctl --user restart $SERVICE
sleep 2
systemctl --user status $SERVICE --no-pager -l
REMOTE

echo ""
echo "Dashboard: http://100.110.98.72:5050/"
echo "           (accessible from any Tailscale-connected device)"
