#!/usr/bin/env bash
set -euo pipefail

install_dir="${1:-/opt/naqafin-caption-worker}"
service_name="naqafin-caption-worker.service"

sudo mkdir -p "$install_dir"
sudo rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'storage/*' \
  --exclude 'models/*' \
  ./ "$install_dir/"

if [[ ! -f "$install_dir/.env" ]]; then
  sudo cp "$install_dir/.env.example" "$install_dir/.env"
fi

sudo cp "systemd/$service_name" "/etc/systemd/system/$service_name"
sudo systemctl daemon-reload
sudo systemctl enable "$service_name"

echo "Installed $service_name. Edit $install_dir/.env, then run:"
echo "  sudo systemctl start $service_name"
