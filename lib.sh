#!/usr/bin/env bash
# Shared helpers for start.sh, stop.sh and status.sh. Sourced, not run.
# Everything executes from this machine over SSH; nothing is installed or copied onto the hosts.
# Written for bash 3.2 (macOS default).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Site settings: cluster.env (copied from cluster.env.example), or CLUSTER_ENV=/path/to/file.
CLUSTER_ENV="${CLUSTER_ENV:-$SCRIPT_DIR/cluster.env}"
if [[ ! -f "$CLUSTER_ENV" ]]; then
  echo "Cluster config not found: $CLUSTER_ENV" >&2
  echo "Create it with: cp cluster.env.example cluster.env   (then set HEAD_HOST, WORKER_HOST, HEAD_IP, WORKER_IP)" >&2
  exit 1
fi
# Node names and IPs from the environment win over the file, e.g. HEAD_HOST=spark-01 ./status.sh
for _v in HEAD_HOST WORKER_HOST HEAD_IP WORKER_IP; do eval "_env_$_v=\${$_v-}"; done
# shellcheck source=cluster.env.example
source "$CLUSTER_ENV"
for _v in HEAD_HOST WORKER_HOST HEAD_IP WORKER_IP; do
  eval "if [[ -n \"\$_env_$_v\" ]]; then $_v=\$_env_$_v; fi"
done
# network.sh narrows this list because it runs before the IPs are known.
for _v in ${LIB_REQUIRE:-HEAD_HOST WORKER_HOST HEAD_IP WORKER_IP}; do
  [[ -n "${!_v:-}" ]] || { echo "$_v is not set in $CLUSTER_ENV" >&2; exit 1; }
done
MODEL_FILE="$SCRIPT_DIR/models/$MODEL.env"
[[ -f "$MODEL_FILE" ]] || { echo "Model file not found: $MODEL_FILE" >&2; exit 1; }
# shellcheck source=/dev/null
source "$MODEL_FILE"

NODES="head worker"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10
  -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/dgx-spark-%C" -o ControlPersist=120)

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m !!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERR\033[0m %s\n' "$*" >&2; exit 1; }

node_host() {
  case "$1" in
    head) echo "$HEAD_HOST" ;;
    worker) echo "$WORKER_HOST" ;;
    *) die "unknown node '$1'" ;;
  esac
}

node_ip() {
  case "$1" in
    head) echo "$HEAD_IP" ;;
    worker) echo "$WORKER_IP" ;;
    *) die "unknown node '$1'" ;;
  esac
}

# rsh HOST CMD ARGS... — run a command on a host with each argument quoted exactly once.
rsh() {
  local host=$1; shift
  # shellcheck disable=SC2029 # arguments are quoted locally on purpose
  ssh -n "${SSH_OPTS[@]}" "$host" "$(printf '%q ' "$@")"
}

# Like rsh, but forwards this script's stdin to the remote command.
rsh_in() {
  local host=$1; shift
  # shellcheck disable=SC2029 # arguments are quoted locally on purpose
  ssh "${SSH_OPTS[@]}" "$host" "$(printf '%q ' "$@")"
}

rdocker()    { local node=$1; shift; rsh "$(node_host "$node")" docker "$@"; }
rdocker_in() { local node=$1; shift; rsh_in "$(node_host "$node")" docker "$@"; }
cexec()      { local node=$1; shift; rdocker "$node" exec "$CONTAINER_NAME" "$@"; }

# cexec_bg NODE 'shell command' — detached inside the container, output to `docker logs`.
cexec_bg() { rdocker "$1" exec -d "$CONTAINER_NAME" bash -c "$2 >> /proc/1/fd/1 2>&1"; }

container_state() {
  local state
  # docker inspect prints an empty line before failing for a missing container.
  state=$(rdocker "$1" inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null) || state=""
  echo "${state:-absent}"
}

require_running() {
  local n
  for n in $NODES; do
    [[ "$(container_state "$n")" == running ]] ||
      die "container $CONTAINER_NAME is not running on $(node_host "$n") — run ./start.sh containers"
  done
}

image_id() { rdocker "$1" image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true; }

# Total GPUs registered in the Ray cluster (empty if Ray is down).
ray_gpus() {
  { cexec head ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 |
      sed -E 's#^[0-9.]+/([0-9]+)(\.[0-9]+)? GPU#\1#'; } || true
}

vllm_pid() {
  cexec head bash -c "ps -eo pid=,args= | awk '/[v]llm serve/ {print \$1; exit}'" 2>/dev/null || true
}

ray_worker_procs() {
  local c
  c=$(cexec "$1" bash -c "ps -eo args= | grep -c '[R]ayWorkerWrapper'" 2>/dev/null || true)
  echo "${c:-0}"
}

health() { rsh "$HEAD_HOST" curl -sf -m 5 "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; }

# poll TIMEOUT CMD... — retry CMD every 3s; non-zero if it never succeeds.
poll() {
  local timeout=$1; shift
  local start=$SECONDS
  until "$@"; do
    (( SECONDS - start >= timeout )) && return 1
    sleep 3
  done
}

wait_for() {
  local timeout=$1 what=$2; shift 2
  poll "$timeout" "$@" || die "timed out after ${timeout}s waiting for $what"
}

prefix() {
  local p=$1 line
  while IFS= read -r line || [[ -n "$line" ]]; do printf '%-9s%s\n' "$p" "$line"; done
}

# on_nodes FUNC — run `FUNC head` and `FUNC worker` in parallel with prefixed output.
on_nodes() {
  local n pids="" rc=0 p
  for n in $NODES; do
    ( set -o pipefail; "$@" "$n" 2>&1 | prefix "[$n]" ) &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || rc=1; done
  return $rc
}
