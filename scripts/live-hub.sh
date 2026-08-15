#!/usr/bin/env bash
# Runs both live-hub suites against a real Tensorhub end to end: mints a token,
# provisions a nonced scratch repo, exports the contract, runs the tests.
#
#   ./scripts/live-hub.sh                 # standing master stack on :31550
#   HUB=http://host:port ./scripts/live-hub.sh
#   REPO=existing-repo ./scripts/live-hub.sh   # reuse a repo instead of making one
#
# Load discipline: this box is shared. The run is niced and capped at two jobs,
# and refuses to start a build while the box is already saturated.
set -euo pipefail
cd "$(dirname "$0")/.."

HUB="${HUB:-http://127.0.0.1:31550}"
ORG="${ORG:-cozy}"
LOGIN="${LOGIN:-cozy}"
PASSWORD="${PASSWORD:-Springtime123!}"
LOAD_CEILING="${LOAD_CEILING:-12}"

load() { cut -d' ' -f1 /proc/loadavg; }

if [ "$(printf '%.0f' "$(load)")" -gt "$LOAD_CEILING" ]; then
  echo "load $(load) is above ${LOAD_CEILING}; refusing to build on a saturated shared box" >&2
  echo "re-run when it settles, or raise LOAD_CEILING deliberately" >&2
  exit 1
fi

echo "==> minting a token against $HUB"
TOKEN=$(curl -fsS -m 60 -X POST "$HUB/api/v1/password/login" \
          -H 'Content-Type: application/json' \
          -d "{\"login\":\"$LOGIN\",\"password\":\"$PASSWORD\"}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
[ -n "$TOKEN" ] || { echo "could not mint a token" >&2; exit 1; }

if [ -n "${REPO:-}" ]; then
  echo "==> reusing repo $ORG/$REPO"
else
  REPO="tensorfs-live-$(date +%s)"
  echo "==> creating scratch repo $ORG/$REPO"
  curl -fsS -m 60 -X POST "$HUB/api/v1/repos" \
       -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
       -d "{\"name\":\"$REPO\"}" >/dev/null
fi

echo "==> confirming the snapshot-sync surface answers"
curl -fsS -m 30 -H "Authorization: Bearer $TOKEN" \
     "$HUB/api/v1/repos/$ORG/$REPO/snapshot-sync/head" >/dev/null

export TENSORFS_HUB_URL="$HUB"
export TENSORFS_HUB_ORG="$ORG"
export TENSORFS_HUB_REPO="$REPO"
export TENSORFS_HUB_TOKEN="$TOKEN"

# One repo, one head: the arms must not race each other.
echo "==> running the live suites (repo $ORG/$REPO)"
nice -n 19 cargo test -j 2 -p tensorfs-core \
     --test live_hub --test live_hub_matrix \
     -- --nocapture --test-threads=1

echo
echo "PASSED. Scratch repo left in place for inspection: $ORG/$REPO"
echo "  delete it with:"
echo "    curl -fsS -X DELETE '$HUB/api/v1/repos/$ORG/$REPO' -H 'Authorization: Bearer \$TOKEN'"
