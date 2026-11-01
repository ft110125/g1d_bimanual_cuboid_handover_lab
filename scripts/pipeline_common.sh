#!/usr/bin/env bash
# Shared helpers for the G1-D demonstration and imitation-learning pipeline.

set -euo pipefail

G1D_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G1D_ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
G1D_ISAAC_PYTHON="${ISAAC_PYTHON:-$G1D_ISAACLAB_ROOT/_isaac_sim/python.sh}"
G1D_DATA_DIR="${G1D_DATA_DIR:-$G1D_PROJECT_ROOT/datasets/g1d_bimanual_cuboid_handover}"
G1D_DEVICE="${G1D_TELEOP_DEVICE:-cuda:0}"

_g1d_die() {
    echo "ERROR: $*" >&2
    exit 1
}

_g1d_is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

g1d_load_project_env() {
    local env_file="$G1D_PROJECT_ROOT/.env.teleop"
    if [[ -f "$env_file" && ! -L "$env_file" ]]; then
        set -a
        # shellcheck disable=SC1090
        . "$env_file"
        set +a
    fi
    G1D_DEVICE="${G1D_TELEOP_DEVICE:-cuda:0}"
    export G1D_DEVICE
}

g1d_require_runtime() {
    [[ -x "$G1D_ISAAC_PYTHON" ]] || _g1d_die "Isaac Sim Python was not found: $G1D_ISAAC_PYTHON"
    [[ -f "$G1D_PROJECT_ROOT/scripts/run_isaaclab_tool.py" ]] || \
        _g1d_die "Project wrapper was not found: $G1D_PROJECT_ROOT/scripts/run_isaaclab_tool.py"
}

g1d_require_file() {
    [[ -f "$1" ]] || _g1d_die "Required file was not found: $1"
}

g1d_require_nonnegative_integer() {
    [[ "$2" =~ ^[0-9]+$ ]] || _g1d_die "$1 must be a non-negative integer: $2"
}

g1d_require_positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || _g1d_die "$1 must be a positive integer: $2"
}

g1d_prepare_output_file() {
    local output="$1"
    mkdir -p "$(dirname "$output")"
    if [[ -e "$output" ]]; then
        if _g1d_is_true "${G1D_OVERWRITE_OUTPUT:-0}"; then
            local backup="${output}.bak.$(date +%Y%m%d_%H%M%S)"
            mv "$output" "$backup"
            echo "Existing output moved to: $backup"
        else
            _g1d_die "Output already exists: $output (set G1D_OVERWRITE_OUTPUT=1 to back it up and continue)"
        fi
    fi
}

g1d_setup_cloudxr() {
    local candidate="${XR_RUNTIME_JSON:-}"
    if [[ -n "$candidate" && -f "$candidate" ]]; then
        :
    else
        candidate=""
        local path
        for path in \
            /usr/share/openxr/1/openxr_cloudxr.json \
            /opt/openxr/share/openxr/1/openxr_cloudxr.json \
            /workspace/cloudxr-runtime_v6.2.1/share/openxr/1/openxr_cloudxr.json; do
            if [[ -f "$path" ]]; then
                candidate="$path"
                break
            fi
        done
        if [[ -z "$candidate" ]]; then
            candidate="$(find /usr /opt /workspace /cloudxr -name openxr_cloudxr.json -type f 2>/dev/null | head -n 1 || true)"
        fi
    fi

    [[ -n "$candidate" && -f "$candidate" ]] || \
        _g1d_die "CloudXR OpenXR runtime configuration openxr_cloudxr.json was not found"

    export XR_RUNTIME_JSON="$candidate"
    export NV_CXR_ENABLE_PUSH_DEVICES=1
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/0}"
    mkdir -p "$XDG_RUNTIME_DIR"
}

g1d_run_isaaclab_tool() {
    local tool="$1"
    shift
    g1d_require_runtime
    g1d_require_file "$tool"
    cd "$G1D_PROJECT_ROOT"
    "$G1D_ISAAC_PYTHON" "$G1D_PROJECT_ROOT/scripts/run_isaaclab_tool.py" "$tool" "$@"
}
