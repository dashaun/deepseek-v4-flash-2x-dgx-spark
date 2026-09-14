#!/usr/bin/env bash
# Show the state of the two-node DGX Spark vLLM cluster.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

node_line() {
  local node=$1 host state started="" avail
  host=$(node_host "$node")
  if ! rsh "$host" true 2>/dev/null; then
    printf '  %-7s %-11s UNREACHABLE\n' "$node" "$host"
    return 0
  fi
  state=$(container_state "$node")
  if [[ "$state" != absent ]]; then
    started=$(rdocker "$node" inspect -f '{{.State.StartedAt}}' "$CONTAINER_NAME" 2>/dev/null | cut -c1-19 | tr T ' ')
  fi
  # shellcheck disable=SC2016 # awk program runs remotely
  avail=$(rsh "$host" awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  printf '  %-7s %-11s %-14s container=%-8s mem_avail=%4sG  %s\n' \
    "$node" "$host" "$(node_ip "$node")" "$state" "$avail" "${started:+since $started UTC}"
}

log "Nodes ($MODEL)"
for n in $NODES; do node_line "$n"; done

log "Ray"
if [[ "$(container_state head)" == running ]]; then
  gpus=$(ray_gpus)
  if [[ -n "$gpus" ]]; then
    active=$(cexec head ray status 2>/dev/null |
      awk '/^Active:/ {f=1; next} /^(Pending|Recent failures):/ {f=0} f && /node_/ {c++} END {print c+0}')
    printf '  %s node(s), %s GPU(s) (want %s)\n' "$active" "$gpus" "$TP_SIZE"
  else
    echo "  not running"
  fi
else
  echo "  head container not running"
fi

log "vLLM"
pid=$(vllm_pid)
if [[ -z "$pid" ]]; then
  echo "  not running"
else
  if health; then
    models=$(rsh "$HEAD_HOST" curl -s -m 5 "http://127.0.0.1:$API_PORT/v1/models" |
      grep -oE '"id":"[^"]*","object":"model"' | cut -d'"' -f4 | tr '\n' ' ')
    printf '  healthy (pid %s) — http://%s:%s/v1  models: %s\n' "$pid" "$HEAD_HOST" "$API_PORT" "$models"
  else
    printf '  starting (pid %s) — last log: %s\n' "$pid" \
      "$(rdocker head logs --tail 1 "$CONTAINER_NAME" 2>&1 | cut -c1-120)"
  fi
fi

echo
echo "  logs:  ssh $HEAD_HOST docker logs -f $CONTAINER_NAME"
