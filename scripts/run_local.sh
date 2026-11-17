#!/usr/bin/env bash
# Run one prediction from local files and save the video next to the audio.
#   usage: scripts/run_local.sh <image> <audio> [prompt] [extra input JSON]
#   e.g.   scripts/run_local.sh me.jpg me.mp3 "A man talks to the camera." '{"sage_attention": true}'
# The files go to Replicate's Files API (private to your account), not anywhere public.
# Needs REPLICATE_API_TOKEN; if it isn't set, it asks for it without echoing it.
# DRY_RUN=1 prints the request instead of sending it. MODEL picks another model.
set -euo pipefail
api=https://api.replicate.com/v1
model="${MODEL:-synthesys-ai/longcat-video-avatar-1.6}"
image="${1:?image path}"
audio="${2:?audio path}"
prompt="${3:-A person talks to the camera, natural expressions and head movement.}"
extra="${4:-}"
[ -n "$extra" ] || extra='{}'
for f in "$image" "$audio"; do [ -f "$f" ] || { echo "No such file: $f"; exit 1; }; done

# Cover the whole recording (the model's limit is 90 s at 480p).
seconds=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$audio" 2>/dev/null || echo 90)
max_duration=$(python3 -c "import math,sys; print(min(90, math.ceil(float(sys.argv[1])) + 1))" "$seconds")

input_json() {  # $1 image url, $2 audio url
  jq -n --arg i "$1" --arg a "$2" --arg p "$prompt" --argjson d "$max_duration" --argjson x "$extra" \
    '{image: $i, audio: $a, prompt: $p, max_duration: $d, seed: 42} + $x'
}

if [ "${DRY_RUN:-}" = 1 ]; then
  echo "model: $model  audio: ${seconds}s -> max_duration $max_duration"
  input_json "<uploaded $(basename "$image")>" "<uploaded $(basename "$audio")>"
  exit 0
fi

if [ -z "${REPLICATE_API_TOKEN:-}" ]; then
  read -rsp "Replicate API token (input hidden): " REPLICATE_API_TOKEN
  echo
fi
auth="Authorization: Bearer ${REPLICATE_API_TOKEN}"

upload() {
  curl -sS --retry 3 -X POST -H "$auth" -F "content=@$1;filename=$(basename "$1")" "$api/files" | jq -r '.urls.get // empty'
}
echo "Uploading $(basename "$image") and $(basename "$audio") to your Replicate account..."
image_url=$(upload "$image"); audio_url=$(upload "$audio")
[ -n "$image_url" ] && [ -n "$audio_url" ] || { echo "Upload failed (check the token)."; exit 1; }

version=$(curl -sS --retry 3 -H "$auth" "$api/models/$model" | jq -r '.latest_version.id // empty')
[ -n "$version" ] || { echo "Could not read the model's latest version."; exit 1; }
body=$(jq -n --arg v "$version" --argjson in "$(input_json "$image_url" "$audio_url")" '{version: $v, input: $in}')
id=$(curl -sS --retry 3 -X POST -H "$auth" -H "Content-Type: application/json" -d "$body" "$api/predictions" | jq -r '.id // empty')
[ -n "$id" ] || { echo "Could not start the prediction."; exit 1; }
echo "Prediction $id (version ${version:0:8}): https://replicate.com/p/$id"

last=""
while :; do
  p=$(curl -sS --retry 3 -H "$auth" "$api/predictions/$id" || true)
  status=$(jq -r '.status // "unknown"' <<<"$p" 2>/dev/null || echo unknown)
  [ "$status" != "$last" ] && echo "$(date +%H:%M:%S) $status" && last=$status
  case "$status" in succeeded|failed|canceled) break ;; esac
  sleep 10
done

jq -r '.logs // ""' <<<"$p" | grep -E '^\[predict\]' || true
if [ "$status" != succeeded ]; then jq '{status, error}' <<<"$p"; exit 1; fi
url=$(jq -r 'if (.output|type) == "array" then .output[0] else .output end' <<<"$p")
out="$(dirname "$audio")/$(basename "${audio%.*}")-longcat.mp4"
curl -sSL -o "$out" "$url"
secs=$(jq -r '.metrics.predict_time' <<<"$p")
python3 -c "import sys; t=float(sys.argv[1]); print(f'GPU time {t:.0f} s, about \${t*0.001525:.2f} on an H100')" "$secs"
echo "Saved $out"
