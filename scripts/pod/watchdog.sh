#!/usr/bin/env bash
# Stop this pod after HOURS no matter what (billing bound for pods kept running between jobs).
#   rp run <pod> --job watchdog -- bash scripts/pod/watchdog.sh 10
HOURS="${1:?hours}"
cd "$(dirname "$0")/../.." || exit 1
set -a; [ -f .env ] && . ./.env; set +a
id="${RUNPOD_POD_ID:-}"
[ -z "$id" ] && [ -r /proc/1/environ ] && id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)
[ -n "$id" ] && [ -n "${RUNPOD_API_KEY:-}" ] || { echo "!! no pod id or RUNPOD_API_KEY: watchdog cannot stop the pod"; exit 1; }
echo "$(date -u +%FT%TZ) watchdog armed: stop pod $id after $HOURS h"
sleep "$(awk "BEGIN{print int($HOURS * 3600)}")"
echo "$(date -u +%FT%TZ) watchdog: stopping pod $id"
curl -s -X POST "https://rest.runpod.io/v1/pods/$id/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
