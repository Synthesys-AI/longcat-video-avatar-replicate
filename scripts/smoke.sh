#!/usr/bin/env bash
# Run (or follow) a test prediction on Replicate and wait for it to finish.
#   usage: scripts/smoke.sh [prediction_id]
#   env:   REPLICATE_API_TOKEN (required), MODEL ("owner/name", needed to start one),
#          VERSION (optional, defaults to the model's latest), WAIT_MINUTES (default 70),
#          SMOKE_SECONDS (max_duration, default 8; the demo speech is 82 s long),
#          SMOKE_SPARSITY (sparse_attention, default unset = the model's default),
#          SMOKE_INPUT (a JSON object merged into the input, e.g. '{"sage_attention": true}')
# Polling tolerates transient HTTP errors; a cold boot downloads ~63 GB of weights first.
set -uo pipefail
api=https://api.replicate.com/v1
auth="Authorization: Bearer ${REPLICATE_API_TOKEN}"
id="${1:-}"

if [ -z "$id" ]; then
  version="${VERSION:-}"
  [ -n "$version" ] || version=$(curl --retry 5 --retry-all-errors --retry-delay 5 -s -H "$auth" "$api/models/${MODEL}" | jq -r '.latest_version.id // empty')
  [ -n "$version" ] || { echo "No version found for ${MODEL}"; exit 1; }
  assets="https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/58eb336edc5c8c63389838727e40b8c18434e54e/assets/avatar/single"
  extra="${SMOKE_INPUT:-}"
  [ -n "$extra" ] || extra='{}'
  body=$(jq -n --arg v "$version" --arg a "$assets" --arg d "${SMOKE_SECONDS:-8}" --arg s "${SMOKE_SPARSITY:-}" --argjson x "$extra" '{
    version: $v,
    input: {
      audio: ($a + "/man.mp3"),
      image: ($a + "/man.png"),
      prompt: "A man stands on stage under dramatic lighting, holding a microphone close to his mouth, speaking while smoke swirls around him.",
      max_duration: ($d | tonumber),
      seed: 42
    }
  } | if $s != "" then .input.sparse_attention = ($s | tonumber) else . end | .input += $x')
  id=$(curl --retry 5 --retry-all-errors --retry-delay 5 -s -X POST -H "$auth" -H "Content-Type: application/json" -d "$body" "$api/predictions" | jq -r '.id // empty')
  [ -n "$id" ] || { echo "Could not start a prediction"; exit 1; }
fi
echo "Prediction $id: https://replicate.com/p/$id"

deadline=$((SECONDS + ${WAIT_MINUTES:-70} * 60))
status=unknown
last=""
while [ "$SECONDS" -lt "$deadline" ]; do
  code=$(curl --retry 5 --retry-all-errors --retry-delay 5 -s -o p.json -w '%{http_code}' -H "$auth" "$api/predictions/$id")
  if [ "$code" = "200" ]; then
    status=$(jq -r .status p.json)
    [ "$status" != "$last" ] && echo "$(date -u +%H:%M:%S) $status"
    last=$status
    case "$status" in succeeded|failed|canceled) break ;; esac
  else
    echo "$(date -u +%H:%M:%S) poll got HTTP $code, retrying"
  fi
  sleep 20
done

echo "----- logs (tail) -----"
jq -r '.logs // ""' p.json 2>/dev/null | tail -80
jq '{status, output, error, metrics}' p.json 2>/dev/null
[ "$status" = "succeeded" ]
