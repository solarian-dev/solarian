#!/usr/bin/env python3
"""CPU of a running ``solarian run-robot``, per process and per thread, from /proc.

Linux only, standard library only. Samples every process under the run-robot
parent twice, ``--seconds`` apart, and prints each one's share of one core,
with its context switches. A process that burns CPU with almost no voluntary
switches is spinning rather than waiting.

    python scripts/robot_cpu.py                 # 30 s window
    python scripts/robot_cpu.py -s 10 --threads
    python scripts/robot_cpu.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _stat(path: str) -> tuple[str, int, int] | None:
    """``(comm, ppid, utime+stime ticks)`` from a ``stat`` file."""
    text = _read(path)
    if text is None:
        return None
    comm = text[text.index("(") + 1 : text.rindex(")")]
    rest = text[text.rindex(")") + 2 :].split()
    return comm, int(rest[1]), int(rest[11]) + int(rest[12])


def _switches(path: str) -> tuple[int, int]:
    voluntary = involuntary = 0
    for line in (_read(path) or "").splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            voluntary = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            involuntary = int(line.split()[1])
    return voluntary, involuntary


def _cmdline(pid: int) -> str:
    return (_read(f"/proc/{pid}/cmdline") or "").replace("\0", " ").strip()


def _all_pids() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def _ancestors(pid: int) -> set[int]:
    out = set()
    while pid > 1:
        out.add(pid)
        st = _stat(f"/proc/{pid}/stat")
        if st is None:
            break
        pid = st[1]
    return out


def find_root(pattern: str) -> int | None:
    """The run-robot process whose parent is not one too.

    This process and the shells that started it are skipped: their command
    lines carry the pattern as an argument.
    """
    matches = {pid for pid in _all_pids() if pattern in _cmdline(pid)}
    matches -= _ancestors(os.getpid())
    for pid in sorted(matches):
        st = _stat(f"/proc/{pid}/stat")
        if st is not None and st[1] not in matches:
            return pid
    return None


def tree(root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid in _all_pids():
        st = _stat(f"/proc/{pid}/stat")
        if st is not None:
            children.setdefault(st[1], []).append(pid)
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(children.get(pid, []))
    return out


def snapshot(pids: list[int]) -> dict:
    snap = {}
    for pid in pids:
        st = _stat(f"/proc/{pid}/stat")
        if st is None:
            continue
        threads = {}
        for tid in os.listdir(f"/proc/{pid}/task") if os.path.isdir(f"/proc/{pid}/task") else []:
            tst = _stat(f"/proc/{pid}/task/{tid}/stat")
            if tst is not None:
                threads[int(tid)] = (tst[0], tst[2])
        snap[pid] = {
            "comm": st[0],
            "ticks": st[2],
            "switches": _switches(f"/proc/{pid}/status"),
            "threads": threads,
        }
    return snap


def system_ticks() -> tuple[int, int]:
    """``(busy, total)`` jiffies over all cores, from /proc/stat."""
    fields = [int(v) for v in (_read("/proc/stat") or "").splitlines()[0].split()[1:]]
    idle = fields[3] + fields[4]
    total = sum(fields[:8])
    return total - idle, total


def cpu_freq_mhz() -> float | None:
    freqs = []
    for cpu in range(os.cpu_count() or 1):
        text = _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        if text:
            freqs.append(int(text) / 1000.0)
    return sum(freqs) / len(freqs) if freqs else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-s", "--seconds", type=float, default=30.0)
    parser.add_argument("-p", "--pattern", default="run-robot", help="cmdline substring of the root")
    parser.add_argument("--pid", type=int, help="root pid, instead of finding it by --pattern")
    parser.add_argument("--threads", action="store_true", help="also list threads using over 1%% of a core")
    parser.add_argument("--json", help="also write the result here")
    args = parser.parse_args()

    root = args.pid if args.pid is not None else find_root(args.pattern)
    if root is None:
        print(f"No process matching {args.pattern!r}", file=sys.stderr)
        return 1
    pids = tree(root)
    sys0 = system_ticks()
    t0 = time.monotonic()
    before = snapshot(pids)
    time.sleep(args.seconds)
    after = snapshot(pids)
    elapsed = time.monotonic() - t0
    sys1 = system_ticks()

    def pct(ticks: int) -> float:
        return 100.0 * ticks / CLK_TCK / elapsed

    rows = []
    for pid, a in after.items():
        b = before.get(pid)
        if b is None:
            continue
        threads = []
        for tid, (comm, ticks) in a["threads"].items():
            prev = b["threads"].get(tid)
            if prev is not None:
                threads.append({"tid": tid, "comm": comm, "cpu": pct(ticks - prev[1])})
        threads.sort(key=lambda t: -t["cpu"])
        rows.append(
            {
                "pid": pid,
                "name": a["comm"],
                "cpu": pct(a["ticks"] - b["ticks"]),
                "voluntary_per_s": (a["switches"][0] - b["switches"][0]) / elapsed,
                "involuntary_per_s": (a["switches"][1] - b["switches"][1]) / elapsed,
                "threads": threads,
            }
        )
    rows.sort(key=lambda r: -r["cpu"])

    cores = os.cpu_count() or 1
    tree_cpu = sum(r["cpu"] for r in rows)
    busy, total = sys1[0] - sys0[0], sys1[1] - sys0[1]
    system_pct = 100.0 * busy / total if total else 0.0
    governor = (_read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "?").strip()
    freq = cpu_freq_mhz()

    print(f"run-robot pid {root}, {elapsed:.1f} s, {cores} cores, governor {governor}"
          + (f", {freq:.0f} MHz now" if freq else ""))
    print(f"{'process':18s} {'pid':>7s} {'% core':>8s} {'vol/s':>8s} {'invol/s':>8s}")
    for r in rows:
        print(f"{r['name']:18s} {r['pid']:7d} {r['cpu']:8.1f} {r['voluntary_per_s']:8.0f} {r['involuntary_per_s']:8.0f}")
        if args.threads:
            for t in r["threads"]:
                if t["cpu"] >= 1.0:
                    print(f"    thread {t['tid']:<9d} {t['comm']:18s} {t['cpu']:6.1f}")
    print(f"{'run-robot total':18s} {'':7s} {tree_cpu:8.1f}   = {tree_cpu / cores:.1f}% of all cores")
    print(f"{'whole system':18s} {'':7s} {system_pct * cores:8.1f}   = {system_pct:.1f}% of all cores")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "root": root,
                    "seconds": elapsed,
                    "cores": cores,
                    "governor": governor,
                    "processes": rows,
                    "tree_pct_of_one_core": tree_cpu,
                    "system_pct_of_all_cores": system_pct,
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
