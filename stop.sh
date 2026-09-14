#!/usr/bin/env bash
# Stop vLLM on the two-node DGX Spark cluster.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<EOF
Usage: ./stop.sh [all|serve]

  all    (default) stop vLLM, then remove $CONTAINER_NAME (and Ray with it) on both nodes
  serve  stop only vLLM; containers, Ray and mods stay up so ./start.sh serve relaunches quickly
EOF
}

no_vllm() { [[ -z "$(vllm_pid)" ]]; }
no_ray_workers() { [[ "$(ray_worker_procs "$1")" -eq 0 ]]; }

stop_serve() {
  local pid n
  if [[ "$(container_state head)" != running ]]; then
    ok "head container not running"
    return 0
  fi

  pid=$(vllm_pid)
  if [[ -z "$pid" ]]; then
    ok "vLLM is not running"
  else
    log "Stopping vLLM on $HEAD_HOST (pid $pid)"
    cexec head kill -INT "$pid" || true
    if ! poll 120 no_vllm; then
      warn "vLLM still running after 120s; sending SIGKILL"
      cexec head kill -KILL "$pid" || true
      poll 15 no_vllm || die "could not stop vLLM (pid $pid)"
    fi
    ok "vLLM stopped"
  fi

  # Ray tears down the TP workers once the driver exits; make sure GPU memory is released.
  for n in $NODES; do
    [[ "$(container_state "$n")" == running ]] || continue
    if ! poll 60 no_ray_workers "$n"; then
      warn "Ray workers still alive on $(node_host "$n"); killing them"
      cexec "$n" pkill -KILL -f RayWorkerWrapper || true
    fi
  done
}

remove_container() {
  local node=$1 host
  host=$(node_host "$node")
  if [[ "$(container_state "$node")" == absent ]]; then
    ok "no $CONTAINER_NAME on $host"
    return 0
  fi
  rdocker "$node" rm -f "$CONTAINER_NAME" >/dev/null
  ok "removed $CONTAINER_NAME on $host"
}

case "${1:-all}" in
  all)
    stop_serve || warn "clean vLLM shutdown failed; removing containers anyway"
    log "Removing containers"
    on_nodes remove_container
    ;;
  serve) stop_serve ;;
  -h | --help | help) usage ;;
  *) usage >&2; exit 1 ;;
esac
