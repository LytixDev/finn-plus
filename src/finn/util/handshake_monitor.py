"""
NOTE: Large chunks of this file was heavily written by an LLM.

AXI-Stream handshake instrumentation for FINN rtlsim.

After several failed attempts using SystemVerilog bind constructs we (Nicolai and his LLM) landed on this design that modifies finn_design_wrapper.v to expose internal handshake signals.

Adds:
- 2 new wide output ports (one per BD scope) carrying packed (valid, ready) pairs for every inter-operator stream in that scope.
- 2 continuous assign statements driving those outputs via hierarchical references into the relevant internal scopes.

New ports become top-level signals that are read from Python every cycle using top.getPort()

Packing:
    For stream i in a scope, bits [2i+1 : 2i] = (valid, ready).
    State = (valid << 1) | ready:
        0 = idle, 1 = starved, 2 = backpressured, 3 = transfer
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass


# Wrapper modules whose inter-operator wires we want to observe, plus the
# Verilog hierarchical-reference prefix from inside `finn_design_wrapper`'s
# body to reach each scope's wires. These prefixes are determined by the
# Vivado BD generation and have been verified for L4 / MLO builds:
#
#   finn_design_wrapper
#     -> finn_design          as finn_design_i
#        -> finn_design_FINNLoop_0_0      as FINNLoop_0      (IP wrapper)
#           -> FINNLoop_0_bd_design       as inst            (BD inside IP)
#              -> FINNLoop_0_imp_*        as FINNLoop_0
#                 -> finn_design_mlo_0    as finn_design_mlo (IP wrapper)
#                    -> finn_design_mlo   as inst            (BD inside IP)
KNOWN_SCOPES: dict[str, str] = {
    "finn_design":     "finn_design_i",
    "finn_design_mlo": "finn_design_i.FINNLoop_0.inst.FINNLoop_0.finn_design_mlo.inst",
}

_VALID_SUFFIXES = ("_TVALID", "_tvalid")
_READY_SUFFIXES = ("_TREADY", "_tready")

_INSTANCE_RE = re.compile(
    r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*\(",
    re.MULTILINE,
)
_PORT_BIND_RE = re.compile(
    r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)


# ---------------------------------------------------------------------------
# Stream discovery (static parsing of BD wrapper Verilog)
# ---------------------------------------------------------------------------

@dataclass
class StreamEntry:
    level_module: str
    producer: str
    producer_port: str
    consumers: list[tuple[str, str]]
    valid_wire: str
    ready_wire: str


def _extract_module_body(verilog_src: str, module_name: str) -> str | None:
    pattern = re.compile(
        rf"^module\s+{re.escape(module_name)}\b(.*?)^endmodule",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(verilog_src)
    return m.group(1) if m else None


def _parse_instances(module_body: str) -> list[dict]:
    instances = []
    for m in _INSTANCE_RE.finditer(module_body):
        module_type, instance_name = m.group(1), m.group(2)
        if module_type in ("wire", "reg", "input", "output", "inout",
                           "assign", "module", "endmodule"):
            continue
        start = m.end()
        depth = 1
        i = start
        while i < len(module_body) and depth > 0:
            c = module_body[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        body = module_body[start:i - 1]
        port_map = {p: w for p, w in _PORT_BIND_RE.findall(body)}
        instances.append({"module": module_type,
                          "instance": instance_name,
                          "ports": port_map})
    return instances


def _classify_handshake_ports(port_map: dict[str, str]) -> dict[str, dict]:
    streams: dict[str, dict] = {}

    def add(stream, role, kind, wire):
        s = streams.setdefault(stream, {"role": role})
        s[f"{kind}_wire"] = wire

    for port, wire in port_map.items():
        for suf in _VALID_SUFFIXES:
            if port.endswith(suf):
                stream = port[: -len(suf)]
                role = ("in" if stream.lower().startswith("in")
                        else "out" if stream.lower().startswith("out") else "?")
                add(stream, role, "valid", wire)
                break
        for suf in _READY_SUFFIXES:
            if port.endswith(suf):
                stream = port[: -len(suf)]
                role = ("in" if stream.lower().startswith("in")
                        else "out" if stream.lower().startswith("out") else "?")
                add(stream, role, "ready", wire)
                break

    return {s: v for s, v in streams.items()
            if "valid_wire" in v and "ready_wire" in v}


def _discover_in_wrapper(verilog_path: str, module_name: str) -> list[StreamEntry]:
    with open(verilog_path) as f:
        src = f.read()
    body = _extract_module_body(src, module_name)
    if body is None:
        return []
    instances = _parse_instances(body)

    wire_users: dict[str, list[tuple[str, str, str]]] = {}
    for inst in instances:
        streams = _classify_handshake_ports(inst["ports"])
        inst["_streams"] = streams
        for stream_name, info in streams.items():
            wire_users.setdefault(info["valid_wire"], []).append(
                (inst["instance"], stream_name, info["role"])
            )

    entries: list[StreamEntry] = []
    for inst in instances:
        for stream_name, info in inst["_streams"].items():
            if info["role"] != "out":
                continue
            consumers = [
                (oi, op)
                for (oi, op, role) in wire_users.get(info["valid_wire"], [])
                if role == "in" and oi != inst["instance"]
            ]
            entries.append(StreamEntry(
                level_module=module_name,
                producer=inst["instance"],
                producer_port=stream_name,
                consumers=consumers,
                valid_wire=info["valid_wire"],
                ready_wire=info["ready_wire"],
            ))
    return entries


def _find_wrapper_source(verilog_srcs: list[str], module_name: str) -> str | None:
    basename_target = f"{module_name}.v"
    candidates = [p for p in verilog_srcs if os.path.basename(p) == basename_target]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if "/ip/src/" in p else 1, p))
    return candidates[0]


def _stream_name(entry: StreamEntry) -> str:
    return f"{entry.level_module}__{entry.producer}__{entry.producer_port}"


def discover_streams_per_scope(verilog_srcs: list[str]) -> dict[str, list[StreamEntry]]:
    """Return {scope_module: [StreamEntry, ...]} for each known scope present."""
    by_scope: dict[str, list[StreamEntry]] = {}
    for module in KNOWN_SCOPES:
        src = _find_wrapper_source(verilog_srcs, module)
        if src is None:
            continue
        streams = _discover_in_wrapper(src, module)
        if streams:
            by_scope[module] = streams
    return by_scope


# ---------------------------------------------------------------------------
# Wrapper-file modification
# ---------------------------------------------------------------------------

INJECT_SENTINEL = "// AUTO-INJECTED-HANDSHAKE-OBSERVER (finn handshake_monitor)"


def _inject_into_wrapper(
    pristine_src: str,
    new_ports: list[tuple[str, int]],
    new_assign_lines: list[str],
) -> str:
    """Return modified wrapper source with new ports + assigns inserted.

    `pristine_src` must be the unmodified original wrapper. We do not attempt
    to detect and strip prior injections; callers are responsible for keeping
    a .pristine backup and always re-injecting from it.

    The wrapper uses old-style Verilog: port-list-in-parens followed by
    separate `input`/`output` declarations and a module body.
    """
    lines = pristine_src.split("\n")

    # 1. Find the line that closes the port list: it's the first line ending
    # in ");" after the `module finn_design_wrapper` line.
    try:
        mod_idx = next(
            i for i, l in enumerate(lines) if "module finn_design_wrapper" in l
        )
    except StopIteration:
        raise RuntimeError("module finn_design_wrapper not found in wrapper")

    plist_end_idx = None
    for i in range(mod_idx, len(lines)):
        s = lines[i].rstrip()
        if s.endswith(");"):
            plist_end_idx = i
            break
    if plist_end_idx is None:
        raise RuntimeError("port list closing `);` not found")

    # Add new port names to the list. Strip the trailing ");" and re-append.
    old = lines[plist_end_idx]
    stripped = old.rstrip()
    assert stripped.endswith(");")
    base = stripped[:-2]  # drop ");"
    new_port_lines = ",\n    ".join(p for p, _ in new_ports)
    lines[plist_end_idx] = base + ",\n    " + new_port_lines + ");"

    # 2. Insert output type declarations. We put them right before the first
    # `wire ` declaration (which always follows the input/output decls), or
    # before the first instantiation line as fallback.
    insert_after_idx = plist_end_idx
    for i in range(plist_end_idx + 1, len(lines)):
        s = lines[i].lstrip()
        if s.startswith(("input ", "output ", "inout ")):
            insert_after_idx = i
    decl_lines = [
        INJECT_SENTINEL,
    ] + [f"  output [{w-1}:0] {n};" for n, w in new_ports] + [
        "  // /AUTO-INJECTED-HANDSHAKE-OBSERVER",
    ]
    lines = lines[: insert_after_idx + 1] + decl_lines + lines[insert_after_idx + 1:]

    # 3. Insert assigns before `endmodule`.
    try:
        endm_idx = next(i for i, l in enumerate(lines) if l.strip() == "endmodule")
    except StopIteration:
        raise RuntimeError("endmodule not found in wrapper")
    assign_block = [INJECT_SENTINEL] + new_assign_lines + [
        "  // /AUTO-INJECTED-HANDSHAKE-OBSERVER",
    ]
    lines = lines[:endm_idx] + assign_block + lines[endm_idx:]

    return "\n".join(lines)


def _build_concat_assign(port_name: str, prefix: str, streams: list[StreamEntry]) -> str:
    """Generate `assign hs_X = {high_stream_valid, high_stream_ready, ...};`.

    Bit layout: stream i occupies bits [2i+1 : 2i] = {valid, ready}.
    SV concat `{a, b}` puts `a` in higher bits, so we list streams highest-first.
    """
    items = []
    for s in reversed(streams):
        items.append(f"        {prefix}.{s.valid_wire}, {prefix}.{s.ready_wire}")
    concat = ",\n".join(items)
    return f"  assign {port_name} = {{\n{concat}\n  }};"


def install_for_model(
    verilog_srcs: list[str],
    wrapper_path: str,
    output_dir: str,
) -> tuple[str, str]:
    """Modify finn_design_wrapper.v in place (with .pristine backup) and emit
    a manifest JSON describing the injected ports + per-stream bit indices.

    Returns (manifest_path, csv_path) where csv_path is where HandshakeMonitorTask
    should write its summary at sim end.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Snapshot the wrapper on first run; always inject from the pristine copy.
    pristine_path = wrapper_path + ".pristine"
    if not os.path.exists(pristine_path):
        shutil.copy2(wrapper_path, pristine_path)
    with open(pristine_path) as f:
        pristine_src = f.read()

    # 2. Discover all inter-operator streams in each known scope.
    by_scope = discover_streams_per_scope(verilog_srcs)
    if not by_scope:
        raise RuntimeError("no handshake scopes discovered")

    # 3. For each scope, build:
    #    - the new output port (name, width)
    #    - the assign-concat line
    #    - the manifest entry
    new_ports: list[tuple[str, int]] = []
    new_assigns: list[str] = []
    manifest_scopes: list[dict] = []

    for scope_module, streams in by_scope.items():
        prefix = KNOWN_SCOPES.get(scope_module)
        if prefix is None:
            continue
        port_name = f"hs_{scope_module}"
        width = 2 * len(streams)
        new_ports.append((port_name, width))
        new_assigns.append(_build_concat_assign(port_name, prefix, streams))
        manifest_scopes.append({
            "scope": scope_module,
            "port": port_name,
            "width": width,
            "hier_prefix": prefix,
            "streams": [
                {
                    "index": i,
                    "name": _stream_name(s),
                    "level": s.level_module,
                    "producer": s.producer,
                    "producer_port": s.producer_port,
                    "valid_wire": s.valid_wire,
                    "ready_wire": s.ready_wire,
                }
                for i, s in enumerate(streams)
            ],
        })

    # 4. Inject into the wrapper.
    modified = _inject_into_wrapper(pristine_src, new_ports, new_assigns)
    with open(wrapper_path, "w") as f:
        f.write(modified)

    # 5. Write manifest + reserve CSV path.
    manifest_path = os.path.abspath(os.path.join(output_dir, "handshake_signals.json"))
    csv_path = os.path.abspath(os.path.join(output_dir, "handshake_summary.csv"))
    with open(manifest_path, "w") as f:
        json.dump({
            "wrapper_path": wrapper_path,
            "pristine_path": pristine_path,
            "scopes": manifest_scopes,
        }, f, indent=2)

    return manifest_path, csv_path


# ---------------------------------------------------------------------------
# Python-side monitoring task (runs inside SimEngine)
# ---------------------------------------------------------------------------

class HandshakeMonitorTask:
    """SimEngine task that reads the injected wrapper ports each cycle and
    accumulates per-stream 4-state counters in Python.

    Designed to be enlisted on the SimEngine like any other task. Behaves as
    a "weak" task (`__bool__ == False`) so it doesn't keep the sim alive on
    its own.
    """

    def __init__(self, sim, manifest_path: str):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.manifest_path = manifest_path
        self.scopes = []  # list of (port_handle, streams_list, counters)
        for scope in manifest["scopes"]:
            port = sim.top.getPort(scope["port"])
            if port is None:
                raise RuntimeError(
                    f"handshake observer port '{scope['port']}' not found "
                    "on top — wrapper was not modified correctly"
                )
            n = len(scope["streams"])
            # Counter layout: [idle, starved, backpressured, xfer] per stream.
            counters = [[0, 0, 0, 0] for _ in range(n)]
            self.scopes.append((scope, port, counters))
        self.ticks = 0

    def __call__(self, sim):
        self.ticks += 1
        for scope, port, counters in self.scopes:
            # Read the wide port value as a hex string and parse to big int.
            hex_str = port.read().as_hexstr()
            try:
                val = int(hex_str, 16) if hex_str else 0
            except ValueError:
                # 'x' or 'z' in the hex string -> treat as 0 for stat purposes.
                val = 0
            for i in range(len(counters)):
                state = (val >> (2 * i)) & 0b11
                counters[i][state] += 1
        return {}

    def __bool__(self):
        return False  # weak task

    def save_csv(self, csv_path: str) -> None:
        cols = ["stream", "cycles", "xfer", "starved", "backpressured", "idle",
                "xfer_frac", "starved_frac", "backpressured_frac", "idle_frac"]
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            rows = []
            for scope, port, counters in self.scopes:
                for i, s in enumerate(scope["streams"]):
                    idle, starved, bp, xfer = counters[i]
                    n = idle + starved + bp + xfer
                    denom = n if n else 1
                    rows.append({
                        "stream": s["name"],
                        "cycles": n,
                        "xfer": xfer,
                        "starved": starved,
                        "backpressured": bp,
                        "idle": idle,
                        "xfer_frac": xfer / denom,
                        "starved_frac": starved / denom,
                        "backpressured_frac": bp / denom,
                        "idle_frac": idle / denom,
                    })
            rows.sort(key=lambda r: -r["cycles"])
            for r in rows:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
