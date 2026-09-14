#!/usr/bin/env bash
# Discover and verify the QSFP / RoCE links between the two Sparks.
# Read-only on the hosts (sysfs, `ip`, `nvidia-smi`, jumbo-frame pings). Only cluster.env is written.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./network.sh [verify | discover [--dry-run]]

  verify     (default) check every cabled link (speed, MTU, RoCE state, jumbo ping) and the
             network settings in cluster.env (ETH_IF, HEAD_IP, WORKER_IP, IB_HCA, IB_GID_INDEX).
             Exits 1 if a check fails.
  discover   detect the cabled QSFP ports, RoCE devices, IPs and RoCE v2 GID index on both nodes,
             confirm each link with a jumbo ping, and write ETH_IF, IB_HCA, IB_GID_INDEX, HEAD_IP
             and WORKER_IP into cluster.env. --dry-run shows the changes without writing.
             Without a cluster.env yet: HEAD_HOST=<head> WORKER_HOST=<worker> ./network.sh discover

Nothing on the hosts is changed. Static IPs and MTU are configured on the hosts themselves
(see NVIDIA's "Connect two Sparks" playbook).
EOF
}

CMD="${1:-verify}"
DRY_RUN=false
if [[ "${2:-}" == --dry-run ]]; then DRY_RUN=true; fi
case "$CMD" in
  discover | verify) ;;
  -h | --help | help) usage; exit 0 ;;
  *) usage >&2; exit 1 ;;
esac

# Remember hosts given on the command line so discover can record them.
CLI_HEAD_HOST="${HEAD_HOST:-}"
CLI_WORKER_HOST="${WORKER_HOST:-}"
CONFIG_FILE="${CLUSTER_ENV:-$ROOT/cluster.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  if [[ "$CMD" != discover || -z "$CLI_HEAD_HOST" || -z "$CLI_WORKER_HOST" ]]; then
    echo "No cluster config at $CONFIG_FILE. Create one with:" >&2
    echo "  HEAD_HOST=<head> WORKER_HOST=<worker> ./network.sh discover" >&2
    exit 1
  fi
  if [[ "$DRY_RUN" == true ]]; then
    export CLUSTER_ENV="$ROOT/cluster.env.example"
  else
    cp "$ROOT/cluster.env.example" "$CONFIG_FILE"
    echo "Created $CONFIG_FILE from cluster.env.example"
    export CLUSTER_ENV="$CONFIG_FILE"
  fi
fi

# The IPs may not be known yet; that's what discover is for.
LIB_REQUIRE="HEAD_HOST WORKER_HOST"
# shellcheck source=lib.sh
source "$ROOT/lib.sh"

# Runs on each Spark via `bash -s`. One line per RoCE device:
#   DEV rdma_dev netdev pci phys_port state operstate speed_mbps mtu ipv4/len roce_v2_ipv4_gid_index
# then: SYS kernel nvidia_driver connectx_fw compaction_proactiveness
read -r -d '' REMOTE_DISCOVER <<'SH' || true
for d in /sys/class/infiniband/*; do
  [ -e "$d" ] || continue
  dev=$(basename "$d")
  net=$(ls "$d/device/net" 2>/dev/null | head -1)
  [ -n "$net" ] || continue
  n=/sys/class/net/$net
  pci=$(basename "$(readlink -f "$d/device")")
  pport=$(cat "$n/phys_port_name" 2>/dev/null); [ -n "$pport" ] || pport=-
  state=$(cut -d' ' -f2 "$d/ports/1/state" 2>/dev/null); [ -n "$state" ] || state=-
  oper=$(cat "$n/operstate" 2>/dev/null || echo -)
  speed=$(cat "$n/speed" 2>/dev/null || echo -)
  mtu=$(cat "$n/mtu" 2>/dev/null || echo 0)
  ip4=$(ip -4 -o addr show dev "$net" 2>/dev/null | awk '{print $4; exit}')
  gid=-
  for t in "$d"/ports/1/gid_attrs/types/*; do
    i=$(basename "$t")
    [ "$(cat "$t" 2>/dev/null)" = "RoCE v2" ] || continue
    case "$(cat "$d/ports/1/gids/$i" 2>/dev/null)" in
      0000:0000:0000:0000:0000:ffff:*) if [ "$gid" = - ] || [ "$i" -lt "$gid" ]; then gid=$i; fi ;;
    esac
  done
  echo "DEV $dev $net $pci $pport $state $oper $speed $mtu ${ip4:--} $gid"
done
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
fw=$(cat /sys/class/infiniband/*/fw_ver 2>/dev/null | sort -u | paste -sd, -)
echo "SYS $(uname -r) ${drv:--} ${fw:--} $(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo -)"
SH

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

NLINKS=0
UNPAIRED=()
FAILS=0
WARNS=0

collect() {
  local n host p pids=""
  for n in $NODES; do
    host=$(node_host "$n")
    printf '%s\n' "$REMOTE_DISCOVER" | rsh_in "$host" bash -s >"$TMP/$n" 2>"$TMP/$n.err" &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || true; done
  for n in $NODES; do
    host=$(node_host "$n")
    grep -q '^SYS ' "$TMP/$n" ||
      die "could not query $host: $(head -1 "$TMP/$n.err") (check HEAD_HOST/WORKER_HOST)"
  done
}

sys_field() { awk -v c="$2" '$1 == "SYS" {print $c; exit}' "$TMP/$1"; }

# shellcheck disable=SC2086 # split the dotted quad on IFS=.
ip_to_int() { local IFS=.; set -- $1; echo $(( ($1 << 24) + ($2 << 16) + ($3 << 8) + $4 )); }

# net_of 192.168.200.1/24 -> 192.168.200.0/24
net_of() {
  local addr=${1%/*} len=${1#*/} n
  n=$(( $(ip_to_int "$addr") & (len == 0 ? 0 : (0xFFFFFFFF << (32 - len)) & 0xFFFFFFFF) ))
  echo "$(( (n >> 24) & 255 )).$(( (n >> 16) & 255 )).$(( (n >> 8) & 255 )).$(( n & 255 ))/$len"
}

worker_line_in_subnet() {
  local target=$1 line ip
  while read -r line; do
    ip=$(echo "$line" | awk '{print $10}')
    if [[ "$ip" != - && "$(net_of "$ip")" == "$target" ]]; then
      echo "$line"
      return 0
    fi
  done < <(grep '^DEV ' "$TMP/worker" || true)
  return 0
}

# Pair head and worker interfaces that share an IPv4 subnet. Links are ordered by PCI address.
# shellcheck disable=SC2034 # every DEV field is named for readability, not all are used
build_links() {
  local dev net pci pport state oper speed mtu ip gid wl matched=" "
  while read -r _ dev net pci pport state oper speed mtu ip gid; do
    if [[ "$oper" != up ]]; then
      UNPAIRED+=("$HEAD_HOST $net ($dev): link $oper (no cable, or peer down)")
      continue
    fi
    if [[ "$ip" == - ]]; then
      UNPAIRED+=("$HEAD_HOST $net ($dev): link up but no IPv4 address")
      continue
    fi
    wl=$(worker_line_in_subnet "$(net_of "$ip")")
    if [[ -z "$wl" ]]; then
      UNPAIRED+=("$HEAD_HOST $net ($dev): no $WORKER_HOST interface on $(net_of "$ip")")
      continue
    fi
    # shellcheck disable=SC2086
    set -- $wl
    L_PORT[NLINKS]=$pport; L_HDEV[NLINKS]=$dev; L_HNET[NLINKS]=$net; L_HSTATE[NLINKS]=$state
    L_HSPEED[NLINKS]=$speed; L_HMTU[NLINKS]=$mtu; L_HIP[NLINKS]=$ip; L_HGID[NLINKS]=$gid
    L_WDEV[NLINKS]=$2; L_WNET[NLINKS]=$3; L_WPORT[NLINKS]=$5; L_WSTATE[NLINKS]=$6; L_WOPER[NLINKS]=$7
    L_WSPEED[NLINKS]=$8; L_WMTU[NLINKS]=$9; L_WIP[NLINKS]=${10}; L_WGID[NLINKS]=${11}
    matched="$matched$3 "
    NLINKS=$((NLINKS + 1))
  done < <(grep '^DEV ' "$TMP/head" | sort -k4,4 || true)

  while read -r _ dev net pci pport state oper speed mtu ip gid; do
    case "$matched" in *" $net "*) continue ;; esac
    if [[ "$oper" != up ]]; then
      UNPAIRED+=("$WORKER_HOST $net ($dev): link $oper (no cable, or peer down)")
    elif [[ "$ip" == - ]]; then
      UNPAIRED+=("$WORKER_HOST $net ($dev): link up but no IPv4 address")
    else
      UNPAIRED+=("$WORKER_HOST $net ($dev): no $HEAD_HOST interface on $(net_of "$ip")")
    fi
  done < <(grep '^DEV ' "$TMP/worker" | sort -k4,4 || true)
}

# Don't-fragment ping at the smaller MTU of each link, from the head over that interface.
ping_links() {
  local i mtu
  for (( i = 0; i < NLINKS; i++ )); do
    mtu=${L_HMTU[i]}
    if (( L_WMTU[i] < mtu )); then mtu=${L_WMTU[i]}; fi
    L_PING[i]=$(rsh "$HEAD_HOST" ping -c 2 -W 2 -M "do" -s "$((mtu - 28))" -I "${L_HNET[i]}" "${L_WIP[i]%/*}" 2>/dev/null |
      awk -F/ '/^(rtt|round-trip)/ {printf "%.2f ms", $5}' || true)
    L_PING[i]=${L_PING[i]:-FAIL}
  done
}

gbps() {
  if [[ "$1" =~ ^[0-9]+$ ]] && (( $1 > 0 )); then echo "$(( $1 / 1000 ))G"; else echo "down"; fi
}

both() { if [[ "$1" == "$2" ]]; then echo "$1"; else echo "$1/$2"; fi; }

print_links() {
  local i n port fmt='  %-7s %-30s %-18s %-30s %-18s %-6s %-9s %-5s %s\n'
  log "QSFP links: $HEAD_HOST (head) <-> $WORKER_HOST (worker)"
  if (( NLINKS == 0 )); then
    echo "  none found"
  else
    # shellcheck disable=SC2059
    printf "$fmt" PORT "HEAD netdev / RoCE" "HEAD IP" "WORKER netdev / RoCE" "WORKER IP" SPEED MTU GID "JUMBO PING"
    for (( i = 0; i < NLINKS; i++ )); do
      port=$(both "${L_PORT[i]}" "${L_WPORT[i]}")
      # shellcheck disable=SC2059
      printf "$fmt" "$port" "${L_HNET[i]} / ${L_HDEV[i]}" "${L_HIP[i]}" "${L_WNET[i]} / ${L_WDEV[i]}" \
        "${L_WIP[i]}" "$(gbps "${L_HSPEED[i]}")" "$(both "${L_HMTU[i]}" "${L_WMTU[i]}")" \
        "$(both "${L_HGID[i]}" "${L_WGID[i]}")" "${L_PING[i]}"
    done
  fi
  for u in ${UNPAIRED[@]+"${UNPAIRED[@]}"}; do echo "  unpaired: $u"; done
  for n in $NODES; do
    printf '  %-11s kernel %s, driver %s, ConnectX fw %s, vm.compaction_proactiveness=%s\n' "$(node_host "$n")" \
      "$(sys_field "$n" 2)" "$(sys_field "$n" 3)" "$(sys_field "$n" 4)" "$(sys_field "$n" 5)"
  done
}

# update_config SRC DST DRY_RUN KEY=VALUE... — rewrite only these keys, keeping trailing comments.
update_config() {
  local src=$1 dst=$2 dry=$3 tmp
  shift 3
  tmp="$TMP/cluster.env.new"
  KV="$(printf '%s\n' "$@")" awk '
    BEGIN {
      n = split(ENVIRON["KV"], pairs, "\n")
      for (i = 1; i <= n; i++) if ((p = index(pairs[i], "=")) > 0) val[substr(pairs[i], 1, p - 1)] = substr(pairs[i], p + 1)
    }
    match($0, /^[A-Za-z_][A-Za-z0-9_]*=/) {
      key = substr($0, 1, RLENGTH - 1)
      if (key in val && !(key in seen)) {
        c = index($0, " #")
        print key "=" val[key] (c ? substr($0, c) : "")
        seen[key] = 1
        next
      }
    }
    { print }
    END { for (k in val) if (!(k in seen)) print k "=" val[k] }' "$src" >"$tmp"

  if cmp -s "$src" "$tmp"; then
    ok "$dst already matches"
    return 0
  fi
  diff -u "$src" "$tmp" | grep -E '^[-+][^-+]' | sed 's/^/  /' || true
  if [[ "$dry" == true ]]; then
    ok "dry run: $dst not written"
  else
    cp "$tmp" "$dst"
    ok "updated $dst"
  fi
}

cmd_discover() {
  collect
  build_links
  ping_links
  print_links
  (( NLINKS > 0 )) ||
    die "no QSFP interface has an IPv4 address on a subnet shared by both nodes — assign static IPs first"

  local i eth="" head_ip="" worker_ip="" hca="" gid=""
  for (( i = 0; i < NLINKS; i++ )); do
    if [[ "${L_PING[i]}" == FAIL ]]; then
      warn "skipping ${L_HNET[i]}: jumbo ping to ${L_WIP[i]%/*} failed (check MTU on both ends)"
      continue
    fi
    if [[ "${L_HSTATE[i]}" != ACTIVE || "${L_WSTATE[i]}" != ACTIVE ]]; then
      warn "skipping ${L_HDEV[i]}: RoCE port is ${L_HSTATE[i]}/${L_WSTATE[i]}"
      continue
    fi
    if [[ "${L_HGID[i]}" == - || "${L_HGID[i]}" != "${L_WGID[i]}" ]]; then
      warn "skipping ${L_HDEV[i]}: no shared RoCE v2 IPv4 GID index (${L_HGID[i]}/${L_WGID[i]})"
      continue
    fi
    if [[ "${L_HDEV[i]}" != "${L_WDEV[i]}" ]]; then
      die "${L_HDEV[i]} on $HEAD_HOST is cabled to ${L_WDEV[i]} on $WORKER_HOST; IB_HCA and ETH_IF apply to both nodes, so connect matching ports or set them by hand"
    fi
    if [[ -z "$gid" ]]; then
      gid=${L_HGID[i]}
    elif [[ "$gid" != "${L_HGID[i]}" ]]; then
      die "RoCE v2 GID index differs between devices ($gid vs ${L_HGID[i]}); set IB_GID_INDEX and IB_HCA by hand"
    fi
    if [[ -z "$eth" ]]; then
      eth=${L_HNET[i]}
      head_ip=${L_HIP[i]%/*}
      worker_ip=${L_WIP[i]%/*}
    fi
    hca="${hca:+$hca,}${L_HDEV[i]}"
  done
  [[ -n "$hca" ]] || die "no usable RoCE link found"

  local values=("HEAD_IP=$head_ip" "WORKER_IP=$worker_ip" "ETH_IF=$eth" "IB_HCA=$hca" "IB_GID_INDEX=$gid")
  if [[ -n "$CLI_HEAD_HOST" ]]; then values+=("HEAD_HOST=$CLI_HEAD_HOST"); fi
  if [[ -n "$CLI_WORKER_HOST" ]]; then values+=("WORKER_HOST=$CLI_WORKER_HOST"); fi

  log "Detected settings"
  printf '  %s\n' "${values[@]}"
  update_config "$CLUSTER_ENV" "$CONFIG_FILE" "$DRY_RUN" "${values[@]}"
  if [[ "$DRY_RUN" != true ]]; then echo "  next: ./network.sh verify"; fi
}

pass()  { printf '  \033[1;32mPASS\033[0m %s\n' "$*"; }
fail()  { printf '  \033[1;31mFAIL\033[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }
check_warn() { printf '  \033[1;33mWARN\033[0m %s\n' "$*"; WARNS=$((WARNS + 1)); }

link_for() { # link_for FIELD VALUE -> index or -1 (FIELD: hnet | hdev)
  local i v
  for (( i = 0; i < NLINKS; i++ )); do
    if [[ "$1" == hnet ]]; then v=${L_HNET[i]}; else v=${L_HDEV[i]}; fi
    if [[ "$v" == "$2" ]]; then echo "$i"; return 0; fi
  done
  echo -1
}

cmd_verify() {
  collect
  build_links
  ping_links
  print_links

  local i li dev name issues used=" " k col label hv wv n v u
  log "Link checks"
  if (( NLINKS == 0 )); then fail "no paired QSFP links"; fi
  for (( i = 0; i < NLINKS; i++ )); do
    name="${L_HNET[i]} <-> ${L_WNET[i]}"
    issues=""
    [[ "${L_WOPER[i]}" == up ]] || issues="$issues; $WORKER_HOST side is ${L_WOPER[i]}"
    [[ "${L_HSPEED[i]}" == "${L_WSPEED[i]}" && "${L_HSPEED[i]}" =~ ^[0-9]+$ ]] ||
      issues="$issues; speed ${L_HSPEED[i]}/${L_WSPEED[i]}"
    [[ "${L_HMTU[i]}" == "${L_WMTU[i]}" ]] || issues="$issues; MTU mismatch ${L_HMTU[i]}/${L_WMTU[i]}"
    [[ "${L_HSTATE[i]}" == ACTIVE && "${L_WSTATE[i]}" == ACTIVE ]] ||
      issues="$issues; RoCE state ${L_HSTATE[i]}/${L_WSTATE[i]}"
    [[ "${L_PING[i]}" != FAIL ]] || issues="$issues; jumbo ping to ${L_WIP[i]%/*} failed"
    if [[ -z "$issues" ]]; then
      pass "$name: $(gbps "${L_HSPEED[i]}"), MTU ${L_HMTU[i]}, RoCE ACTIVE, jumbo ping ${L_PING[i]}"
    else
      fail "$name: ${issues#; }"
    fi
    if [[ "${L_HMTU[i]}" == "${L_WMTU[i]}" && "${L_HMTU[i]}" != 9000 ]]; then
      check_warn "$name: MTU ${L_HMTU[i]} (9000 recommended)"
    fi
    if [[ "${L_HNET[i]}" != "${L_WNET[i]}" ]]; then
      check_warn "$name: cable connects different ports (${L_PORT[i]} to ${L_WPORT[i]})"
    fi
  done
  for u in ${UNPAIRED[@]+"${UNPAIRED[@]}"}; do check_warn "unpaired: $u"; done

  log "cluster.env checks ($CLUSTER_ENV)"
  if [[ -z "${HEAD_IP:-}" || -z "${WORKER_IP:-}" ]]; then
    fail "HEAD_IP/WORKER_IP not set — run ./network.sh discover"
  else
    li=$(link_for hnet "$ETH_IF")
    if (( li < 0 )); then
      fail "ETH_IF=$ETH_IF is not a paired QSFP link"
    elif [[ "${L_WNET[li]}" != "$ETH_IF" || "${L_HIP[li]%/*}" != "$HEAD_IP" || "${L_WIP[li]%/*}" != "$WORKER_IP" ]]; then
      fail "ETH_IF=$ETH_IF links ${L_HIP[li]%/*} to ${L_WNET[li]} ${L_WIP[li]%/*}, but cluster.env has HEAD_IP=$HEAD_IP WORKER_IP=$WORKER_IP"
    else
      pass "ETH_IF=$ETH_IF carries HEAD_IP=$HEAD_IP <-> WORKER_IP=$WORKER_IP"
    fi
  fi

  for dev in ${IB_HCA//,/ }; do
    li=$(link_for hdev "$dev")
    if (( li < 0 )); then
      fail "IB_HCA: $dev is not on a paired QSFP link"
      continue
    fi
    used="$used$dev "
    if [[ "${L_WDEV[li]}" != "$dev" ]]; then
      fail "IB_HCA: $dev is cabled to ${L_WDEV[li]} on $WORKER_HOST (IB_HCA applies to both nodes)"
    elif [[ "${L_HGID[li]}" != "$IB_GID_INDEX" || "${L_WGID[li]}" != "$IB_GID_INDEX" ]]; then
      fail "IB_HCA: $dev RoCE v2 IPv4 GID index is $(both "${L_HGID[li]}" "${L_WGID[li]}"), cluster.env has IB_GID_INDEX=$IB_GID_INDEX"
    else
      pass "IB_HCA: $dev ($(both "${L_PORT[li]}" "${L_WPORT[li]}"), GID $IB_GID_INDEX RoCE v2)"
    fi
  done
  for (( i = 0; i < NLINKS; i++ )); do
    case "$used" in
      *" ${L_HDEV[i]} "*) ;;
      *) check_warn "${L_HDEV[i]} is cabled but not in IB_HCA (unused link capacity)" ;;
    esac
  done

  log "Node checks"
  for k in "2:kernel" "3:NVIDIA driver" "4:ConnectX firmware"; do
    col=${k%%:*}
    label=${k#*:}
    hv=$(sys_field head "$col")
    wv=$(sys_field worker "$col")
    if [[ "$hv" == "$wv" ]]; then pass "$label $hv on both nodes"; else check_warn "$label differs: $hv vs $wv"; fi
  done
  for n in $NODES; do
    v=$(sys_field "$n" 5)
    if [[ "$v" == 0 ]]; then
      pass "$(node_host "$n"): vm.compaction_proactiveness=0"
    else
      check_warn "$(node_host "$n"): vm.compaction_proactiveness=$v (0 recommended with RoCE)"
    fi
  done

  echo
  (( FAILS == 0 )) || die "$FAILS check(s) failed, $WARNS warning(s)"
  ok "all checks passed, $WARNS warning(s)"
}

case "$CMD" in
  discover) cmd_discover ;;
  verify) cmd_verify ;;
esac
