#!/usr/bin/env bash
# Start vLLM on the two-node DGX Spark cluster (settings: cluster.env, models/$MODEL.env).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<EOF
Usage: ./start.sh [step]

Steps (default: all = preflight containers ray mods serve wait):
  preflight   SSH, image digest, model cache, RoCE links, host memory
  containers  start $CONTAINER_NAME on both nodes (idle, sleep infinity)
  ray         start the Ray head, join the worker, wait for $TP_SIZE GPUs
  mods        copy ./mods into both containers and apply them
  nccl-test   all-reduce across both nodes; reports transport (NET/IB) and bus bandwidth
  serve       launch vllm serve $MODEL_ID on the head
  wait        wait for the API to report healthy

Every step is safe to re-run and checks that the previous steps are done.
EOF
}

step_preflight() {
  log "Preflight ($HEAD_HOST head, $WORKER_HOST worker, model $MODEL_ID)"
  on_nodes preflight_node
  local head_img worker_img
  head_img=$(image_id head)
  worker_img=$(image_id worker)
  [[ "$head_img" == "$worker_img" ]] || die "image differs between nodes ($head_img vs $worker_img)"
  rsh "$HEAD_HOST" ping -c 1 -W 2 "$WORKER_IP" >/dev/null ||
    die "$HEAD_HOST cannot reach $WORKER_IP on the QSFP link"
  ok "image identical on both nodes; $HEAD_IP -> $WORKER_IP reachable"
}

preflight_node() {
  local node=$1 host home dir ref dev state avail
  host=$(node_host "$node")
  rsh "$host" true || die "cannot SSH to $host"

  [[ -n "$(image_id "$node")" ]] || die "$IMAGE not present on $host (ssh $host docker pull $IMAGE)"

  home=$(rsh "$host" printenv HOME)
  dir="$home/.cache/huggingface/hub/models--${MODEL_ID//\//--}"
  ref=$(rsh "$host" cat "$dir/refs/main" 2>/dev/null || true)
  [[ -n "$ref" ]] || die "$MODEL_ID not found in the Hugging Face cache on $host"
  if [[ -n "${MODEL_REVISION:-}" && "$ref" != "$MODEL_REVISION" ]]; then
    die "cached $MODEL_ID is at $ref on $host, expected $MODEL_REVISION"
  fi
  rsh "$host" test -d "$dir/snapshots/$ref" || die "snapshot $ref missing on $host"
  if rsh "$host" find "$dir/blobs" -name '*.incomplete' | grep -q .; then
    die "incomplete download of $MODEL_ID on $host"
  fi

  for dev in ${IB_HCA//,/ }; do
    state=$(rsh "$host" cat "/sys/class/infiniband/$dev/ports/1/state" 2>/dev/null || true)
    [[ "$state" == *ACTIVE* ]] || die "RoCE device $dev is not ACTIVE on $host (${state:-missing})"
  done

  # shellcheck disable=SC2016 # awk program runs remotely
  avail=$(rsh "$host" awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  (( avail >= MIN_FREE_GB )) || warn "only ${avail}G available on $host (want ${MIN_FREE_GB}G)"
  [[ "$(rsh "$host" cat /proc/sys/vm/compaction_proactiveness)" == 0 ]] ||
    warn "vm.compaction_proactiveness is not 0 on $host"

  ok "$host: image, model @ ${ref:0:8}, RoCE ${IB_HCA}, ${avail}G available"
}

step_containers() {
  log "Starting containers"
  on_nodes start_container
}

start_container() {
  local node=$1 host ip home state kv
  host=$(node_host "$node")
  ip=$(node_ip "$node")

  state=$(container_state "$node")
  case "$state" in
    running) ok "$CONTAINER_NAME already running on $host"; return 0 ;;
    absent) ;;
    *) die "$CONTAINER_NAME exists on $host in state '$state' — run ./stop.sh first" ;;
  esac

  home=$(rsh "$host" printenv HOME)
  local args=(run -d --name "$CONTAINER_NAME"
    --label "dgx-spark.role=$node" --label "dgx-spark.model=$MODEL"
    --privileged --ipc=host --network host --gpus all
    --ulimit nofile=1048576:1048576 --ulimit memlock=-1
    -v "$home/.cache/huggingface:/root/.cache/huggingface")

  if [[ "$MOUNT_COMPILE_CACHES" == true ]]; then
    rsh "$host" mkdir -p "$home/.cache/vllm" "$home/.cache/flashinfer" "$home/.triton"
    args+=(-v "$home/.cache/vllm:/root/.cache/vllm"
      -v "$home/.cache/flashinfer:/root/.cache/flashinfer"
      -v "$home/.triton:/root/.triton")
  fi

  for kv in \
    "VLLM_HOST_IP=$ip" "RAY_NODE_IP_ADDRESS=$ip" "RAY_OVERRIDE_NODE_IP_ADDRESS=$ip" \
    "NCCL_SOCKET_IFNAME=$ETH_IF" "GLOO_SOCKET_IFNAME=$ETH_IF" "TP_SOCKET_IFNAME=$ETH_IF" \
    "UCX_NET_DEVICES=$ETH_IF" "OMPI_MCA_btl_tcp_if_include=$ETH_IF" "MN_IF_NAME=$ETH_IF" \
    "NCCL_IB_DISABLE=0" "NCCL_IB_HCA=$IB_HCA" "NCCL_IB_GID_INDEX=$IB_GID_INDEX" \
    "NCCL_IGNORE_CPU_AFFINITY=1" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "RAY_memory_monitor_refresh_ms=0" "RAY_num_prestart_python_workers=0" \
    "RAY_object_store_memory=$RAY_OBJECT_STORE_BYTES" \
    ${MODEL_CONTAINER_ENV[@]+"${MODEL_CONTAINER_ENV[@]}"} \
    ${EXTRA_CONTAINER_ENV[@]+"${EXTRA_CONTAINER_ENV[@]}"}; do
    args+=(-e "$kv")
  done
  args+=(--entrypoint sleep "$IMAGE" infinity)

  rdocker "$node" "${args[@]}" >/dev/null
  ok "started $CONTAINER_NAME on $host ($ip)"
}

ray_head_up() { cexec head ray status >/dev/null 2>&1; }
ray_ready()   { [[ "$(ray_gpus)" == "$TP_SIZE" ]]; }
worker_raylet_running() { cexec worker bash -c "ps -eo args= | grep -q '[r]aylet'" 2>/dev/null; }

step_ray() {
  require_running
  if ray_ready; then ok "Ray cluster already up with $TP_SIZE GPUs"; return 0; fi

  if ! ray_head_up; then
    log "Starting Ray head on $HEAD_HOST ($HEAD_IP:$RAY_PORT)"
    cexec_bg head "ray start --block --head --port=$RAY_PORT --node-ip-address=$HEAD_IP \
      --num-cpus=2 --object-store-memory=$RAY_OBJECT_STORE_BYTES \
      --include-dashboard=false --disable-usage-stats"
    wait_for 90 "Ray head" ray_head_up
    ok "Ray head up"
  fi

  if worker_raylet_running; then
    poll 30 ray_ready ||
      die "a raylet is running on $WORKER_HOST but never joined the head — run ./stop.sh && ./start.sh"
  else
    log "Joining Ray worker on $WORKER_HOST ($WORKER_IP)"
    cexec_bg worker "ray start --block --address=$HEAD_IP:$RAY_PORT --node-ip-address=$WORKER_IP \
      --num-cpus=2 --object-store-memory=$RAY_OBJECT_STORE_BYTES --disable-usage-stats"
    wait_for 120 "Ray cluster to report $TP_SIZE GPUs" ray_ready
  fi
  ok "Ray cluster ready: $TP_SIZE GPUs across $HEAD_HOST and $WORKER_HOST"
}

mods_applied() {
  local n mod
  for n in $NODES; do
    for mod in ${MODS[@]+"${MODS[@]}"}; do
      cexec "$n" test -f "/workspace/mods/.applied-$mod" 2>/dev/null || return 1
    done
  done
}

apply_mods_node() {
  local node=$1 mod
  local tar_opts=()
  if tar --version 2>/dev/null | grep -q bsdtar; then tar_opts=(--no-mac-metadata --no-xattrs); fi

  for mod in ${MODS[@]+"${MODS[@]}"}; do
    if cexec "$node" test -f "/workspace/mods/.applied-$mod" 2>/dev/null; then
      ok "$mod already applied"
      continue
    fi
    COPYFILE_DISABLE=1 tar ${tar_opts[@]+"${tar_opts[@]}"} -C "$SCRIPT_DIR/mods" -cf - "$mod" |
      rdocker_in "$node" exec -i "$CONTAINER_NAME" bash -c \
        'mkdir -p /workspace/mods && tar --no-same-owner --warning=no-unknown-keyword -xf - -C /workspace/mods'
    cexec "$node" bash -c "cd /workspace/mods/$mod && bash ./run.sh && touch /workspace/mods/.applied-$mod"
    ok "applied $mod"
  done
}

step_mods() {
  require_running
  local mod
  for mod in ${MODS[@]+"${MODS[@]}"}; do
    [[ -f "$SCRIPT_DIR/mods/$mod/run.sh" ]] || die "mods/$mod/run.sh not found"
  done
  log "Applying mods: ${MODS[*]:-none}"
  on_nodes apply_mods_node
}

read -r -d '' NCCL_TEST_PY <<'PY' || true
import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
x = torch.ones(1, device="cuda")
dist.all_reduce(x)
assert int(x.item()) == world, f"all_reduce returned {x.item()}, expected {world}"
for mib in (64, 256, 1024):
    t = torch.ones(mib * 2**20 // 4, dtype=torch.float32, device="cuda")
    for _ in range(3):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    iters, t0 = 10, time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    busbw = mib * 2**20 * 8 / dt / 1e9 * 2 * (world - 1) / world
    if rank == 0:
        print(f"RESULT all_reduce {mib:5d} MiB  {dt * 1e3:8.1f} ms  busbw {busbw:6.1f} Gbit/s", flush=True)
if rank == 0:
    print("RESULT ok", flush=True)
dist.destroy_process_group()
PY

nccl_rank() {
  local node=$1 rank=$2
  rdocker "$node" exec -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET "$CONTAINER_NAME" \
    python3 -m torch.distributed.run --nnodes=2 --nproc-per-node=1 --node-rank="$rank" \
    --master-addr="$HEAD_IP" --master-port="$NCCL_TEST_PORT" /workspace/nccl_test.py
}

step_nccl_test() {
  require_running
  [[ -z "$(vllm_pid)" ]] || die "vLLM is running and holds GPU memory — run ./stop.sh serve first"

  local n tmp rc=0 wpid
  for n in $NODES; do
    printf '%s\n' "$NCCL_TEST_PY" |
      rdocker_in "$n" exec -i "$CONTAINER_NAME" bash -c 'mkdir -p /workspace && cat > /workspace/nccl_test.py'
  done

  tmp=$(mktemp -d)
  log "NCCL all-reduce test ($HEAD_IP <-> $WORKER_IP, HCAs $IB_HCA)"
  nccl_rank worker 1 >"$tmp/worker.log" 2>&1 &
  wpid=$!
  nccl_rank head 0 >"$tmp/head.log" 2>&1 || rc=1
  wait "$wpid" || rc=1

  grep -h 'RESULT all_reduce' "$tmp/head.log" | sed 's/^RESULT /  /' || true
  echo "  transports:"
  grep -hoE 'via NET/[A-Za-z]+(/[0-9]+)?(/GDRDMA)?' "$tmp/head.log" "$tmp/worker.log" | sort | uniq -c | sed 's/^/  /' || true
  grep -hoE 'NET/IB : Using .*' "$tmp/head.log" | head -1 | sed 's/^/  /' || true

  if (( rc != 0 )) || ! grep -q 'RESULT ok' "$tmp/head.log"; then
    echo "--- head log (tail)"; tail -40 "$tmp/head.log"
    echo "--- worker log (tail)"; tail -40 "$tmp/worker.log"
    die "NCCL test failed (full logs in $tmp)"
  fi
  if grep -q 'via NET/IB' "$tmp/head.log" "$tmp/worker.log"; then
    ok "NCCL is using RDMA (NET/IB)"
  else
    warn "NCCL did not use NET/IB — traffic is going over sockets (logs in $tmp)"
  fi
}

step_serve() {
  require_running
  ray_ready || die "Ray cluster doesn't report $TP_SIZE GPUs — run ./start.sh ray"
  mods_applied || die "mods not applied on both nodes — run ./start.sh mods"
  if [[ -n "$(vllm_pid)" ]]; then ok "vLLM already running on $HEAD_HOST"; return 0; fi

  local cmd=(vllm serve "$MODEL_ID"
    --host 0.0.0.0 --port "$API_PORT"
    --tensor-parallel-size "$TP_SIZE" --distributed-executor-backend ray
    ${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"}
    ${EXTRA_SERVE_ARGS[@]+"${EXTRA_SERVE_ARGS[@]}"})

  # Keep the exact command in the container for debugging: docker exec vllm_node cat /workspace/serve.sh
  {
    echo '#!/bin/bash'
    echo "# generated by start.sh on $(date '+%Y-%m-%d %H:%M:%S') for $MODEL"
    printf 'exec'
    printf ' %q' "${cmd[@]}"
    echo
  } | rdocker_in head exec -i "$CONTAINER_NAME" bash -c 'cat > /workspace/serve.sh && chmod +x /workspace/serve.sh'

  log "Launching vllm serve $MODEL_ID on $HEAD_HOST"
  cexec_bg head /workspace/serve.sh
  sleep 5
  if [[ -z "$(vllm_pid)" ]]; then
    rdocker head logs --tail 40 "$CONTAINER_NAME" 2>&1 || true
    die "vllm serve exited immediately"
  fi
  ok "vllm serve started (logs: ssh $HEAD_HOST docker logs -f $CONTAINER_NAME)"
}

step_wait() {
  local start=$SECONDS line
  log "Waiting for http://$HEAD_HOST:$API_PORT/health (up to ${WAIT_TIMEOUT}s; loading and compiling takes a while)"
  until health; do
    if [[ -z "$(vllm_pid)" ]]; then
      rdocker head logs --tail 60 "$CONTAINER_NAME" 2>&1 || true
      die "vllm serve is no longer running"
    fi
    (( SECONDS - start < WAIT_TIMEOUT )) || die "not healthy after ${WAIT_TIMEOUT}s (vLLM is still running)"
    line=$(rdocker head logs --tail 1 "$CONTAINER_NAME" 2>&1 | cut -c1-140 || true)
    printf '  [%5ss] %s\n' "$((SECONDS - start))" "$line"
    sleep 20
  done
  ok "healthy after $((SECONDS - start))s — API: http://$HEAD_HOST:$API_PORT/v1"
}

case "${1:-all}" in
  all)
    step_preflight
    step_containers
    step_ray
    step_mods
    step_serve
    step_wait
    ;;
  preflight | containers | ray | mods | nccl-test | serve | wait)
    "step_${1//-/_}"
    ;;
  -h | --help | help) usage ;;
  *) usage >&2; exit 1 ;;
esac
