# Live-hub proofs

Two suites run against a **real** Tensorhub carrying th#1960's snapshot-sync routes:

| file | what it proves |
|---|---|
| `live_hub.rs` | the happy path once: push → head → 8-byte-edit dedup → byte-exact pull → zero-byte resume |
| `live_hub_matrix.rs` | repeatability, racing producers, interruption/resume, measured efficiency, the hub's own refusals |

Both are opt-in. Without the env vars they print exactly what is missing and skip.

## Run it

```sh
./scripts/live-hub.sh              # mints a token, makes a scratch repo, runs both suites
```

The script is the whole runbook. What it does, if you need to do it by hand:

```sh
# 1. the standing master stack (AUTHORITATIVE, the integration oracle)
cd ~/cozy/e2e && task stacks          # confirm `master ... http=:31550 ... alive`

# 2. a token — note /api/v1, not /v1; login is slow under load, so give it room
TOKEN=$(curl -fsS -m 60 -X POST http://127.0.0.1:31550/api/v1/password/login \
          -H 'Content-Type: application/json' \
          -d '{"login":"cozy","password":"Springtime123!"}' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

# 3. a scratch repo, so a shared hub's head is never fought over
REPO="tensorfs-live-$(date +%s)"
curl -fsS -X POST http://127.0.0.1:31550/api/v1/repos \
     -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d "{\"name\":\"$REPO\"}"

# 4. run
export TENSORFS_HUB_URL=http://127.0.0.1:31550
export TENSORFS_HUB_ORG=cozy TENSORFS_HUB_REPO="$REPO" TENSORFS_HUB_TOKEN="$TOKEN"
nice -n 19 cargo test -j 2 -p tensorfs-core --test live_hub --test live_hub_matrix -- --nocapture --test-threads=1
```

`--test-threads=1` is required: the arms drive one repo's head and would race each other.

## Knobs

| variable | default | effect |
|---|---|---|
| `TENSORFS_HUB_ROUNDS` | 10 | repeated round trips |
| `TENSORFS_HUB_SCALE_MIB` | 384 | the scale arm's declared payload |
| `TENSORFS_E2E_DIR` | `$TMPDIR/tensorfs-live-e2e` | scratch root |

The scale arm holds its fixture in memory, so `TENSORFS_HUB_SCALE_MIB` is also
roughly its peak RSS. Keep ≥10% of the disc free.

## Teardown

Scratch repos are cheap and nonced, so the suites do not delete them. To clean up:

```sh
curl -fsS -X DELETE "http://127.0.0.1:31550/api/v1/repos/cozy/$REPO" \
     -H "Authorization: Bearer $TOKEN"
```

Never restart, upgrade or GC the standing `master` stack to make a test pass — it
is shared, and other agents' work depends on it.

## Making this a scheduled lane

Today this is a manual gate, so it defends nothing against regressions. To make it
a real one, it needs, in order:

1. **A hub reachable from CI.** The standing stack is a developer box on ngrok; CI
   cannot depend on it. This is the actual blocker — everything below is easy once
   a CI-reachable hub with th#1960's routes exists.
2. **A service token in CI secrets** with `org:repo:write` on one throwaway org, and
   a repo-per-run naming scheme (the suites already nonce their fixtures).
3. **A nightly workflow**, not a required check: the arms move hundreds of MiB and
   are far outside the sub-2-minute budget the required set holds to.
4. **A cleanup step** that deletes run repos, so the hub does not accumulate them.
