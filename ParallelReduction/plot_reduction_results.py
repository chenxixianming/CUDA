"""Plot CUDA reduction logs with automatically detected sizes and block limits.

Install: pip install numpy matplotlib
Run: python plot_reduction_results.py --input reduction_results.txt
Optional: --sizes 1048576 4194304 --blocks 64 256 1024 --out figures

Bars show median batch-average time. Whiskers show min/max batch averages,
not confidence intervals. Missing results are N/A, never zero.
Duplicate configurations use the last record, without pooling separate runs.
"""
from pathlib import Path
import argparse
import math
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np

VERSIONS = ("stride", "shuffle")
COLORS = {"stride": "#476C9B", "shuffle": "#14958A"}
HEADER = re.compile(
    r"^\s*=+\s*N\s*=\s*([\d,]+)\s*,\s*max_blocks\s*=\s*([\d,]+)\s*=+[^\r\n]*$",
    re.MULTILINE | re.IGNORECASE,
)
VERSION = re.compile(r"^\s*Version:\s*(stride|shuffle)\s*$", re.MULTILINE | re.IGNORECASE)


def warn(message):
    warnings.warn(message, UserWarning, stacklevel=2)


def read_log(path):
    data = Path(path).read_bytes()
    # Older Windows PowerShell redirection can produce UTF-16.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("gb18030")
    # Normalize Windows CRLF before applying multiline regular expressions.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_log(path):
    text = read_log(path)
    headers = list(HEADER.finditer(text))
    if not headers:
        raise ValueError("No headers found. Expected: ========== N = ..., max_blocks = ... ==========")
    records = {}
    skipped = 0
    for i, header in enumerate(headers):
        n, cap = (int(s.replace(",", "")) for s in header.groups())
        if n <= 0 or cap <= 0:
            warn(f"Invalid configuration N={n}, max_blocks={cap}; skipped.")
            continue
        end = headers[i+1].start() if i+1 < len(headers) else len(text)
        body = text[header.end():end]
        row = {"N": n, "max_blocks": cap}
        versions = list(VERSION.finditer(body))
        if not versions:
            warn(f"N={n}, max_blocks={cap}: no stride/shuffle Version sections.")
        seen = set()
        for j, match in enumerate(versions):
            name = match.group(1).lower()
            label = f"N={n}, max_blocks={cap}, {name}"
            if name in seen:
                warn(f"{label}: duplicate version; using its last section.")
                row.pop(name, None)
            seen.add(name)
            end = versions[j+1].start() if j+1 < len(versions) else len(body)
            section = body[match.end():end]
            if not re.search(r"^\s*Result:\s*PASS\s*$", section, re.MULTILINE | re.IGNORECASE):
                warn(f"{label}: validation failed or PASS missing; excluded.")
                continue
            if n == 1 or re.search(r"^\s*Rounds:\s*0\s*$", section, re.MULTILINE):
                skipped += 1
                continue
            if re.search(r"Timing skipped:", section, re.IGNORECASE):
                warn(f"{label}: timing skipped; excluded.")
                continue
            lines = re.findall(r"^\s*Batch\s+\d+\s+average:\s*([^\r\n]+)",
                               section, re.MULTILINE | re.IGNORECASE)
            fallback = False
            if not lines:
                lines = re.findall(r"^\s*Median reduction time:\s*([^\r\n]+)",
                                   section, re.MULTILINE | re.IGNORECASE)
                if not lines:
                    lines = re.findall(r"^\s*Average reduction time:\s*([^\r\n]+)",
                                       section, re.MULTILINE | re.IGNORECASE)
                lines = lines[-1:]
                fallback = bool(lines)
            if not lines:
                warn(f"{label}: no timing values; excluded.")
                continue
            try:
                values = []
                for line in lines:
                    m = re.fullmatch(r"\s*(\S+)\s+ms\s*", line, re.IGNORECASE)
                    if m is None:
                        raise ValueError("expected a numeric value followed by ms")
                    value = float(m.group(1))
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("timing must be positive and finite")
                    values.append(value)
            except ValueError as error:
                warn(f"{label}: invalid timing ({error}); excluded.")
                continue
            row[name] = values
            if fallback:
                warn(f"{label}: using reported time; batch variation unavailable.")
        key = (n, cap)
        if key in records:
            warn(f"N={n}, max_blocks={cap}: repeated configuration; using the last record, "
                 "including any missing/failed versions.")
        records[key] = row
    rows = [r for r in records.values() if any(v in r for v in VERSIONS)]
    if skipped:
        print(f"Skipped {skipped} version records with no kernel execution.")
    if not rows:
        raise ValueError("No valid timed results. Check PASS status and positive times. "
                         "A correctness-only log cannot produce timing charts.")
    return sorted(rows, key=lambda r: (r["N"], r["max_blocks"]))


def select_values(requested, available, label):
    available = sorted(set(available))
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        warn(f"Requested {label} not present: {missing}. Available: {available}. Skipping them.")
    selected = [x for x in available if x in requested]
    if not selected:
        raise ValueError(f"No requested {label} available. Available: {available}")
    return selected


def plot(rows, output, sizes=None, caps=None):
    sizes = select_values(sizes, [r["N"] for r in rows], "input sizes")
    candidates = [r for r in rows if r["N"] in sizes]
    caps = select_values(caps, [r["max_blocks"] for r in candidates], "block limits")
    selected = [r for r in candidates if r["max_blocks"] in caps]
    sizes = select_values(sizes, [r["N"] for r in selected], "sizes for selected blocks")
    lookup = {(r["N"], r["max_blocks"]): r for r in selected}
    missing = sum(v not in lookup.get((n,c), {})
                  for n in sizes for c in caps for v in VERSIONS)
    if missing:
        warn(f"{missing} missing configuration/version combinations will appear as N/A.")
    print(f"Input sizes: {sizes}")
    print(f"Block limits: {caps}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#263649", "text.color": "#172D40",
        "xtick.color": "#435568", "ytick.color": "#435568",
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    columns = min(3, len(sizes))
    nrows = math.ceil(len(sizes) / columns)
    panel_width = max(4.7, len(caps) * .85)
    fig, axes = plt.subplots(nrows, columns,
        figsize=(panel_width * columns, 4 * nrows + 1), squeeze=False)
    for ax, n in zip(axes.flat, sizes):
        x = np.arange(len(caps))
        highest = 0
        for k, name in enumerate(VERSIONS):
            for j, cap in enumerate(caps):
                pos = x[j] + (k-.5)*.34
                values = lookup.get((n, cap), {}).get(name)
                if not values:
                    ax.text(pos, .02, "N/A", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", rotation=90, fontsize=9, color="#7E8790")
                    continue
                vals = np.asarray(values, dtype=float)*1000
                median = float(np.median(vals))
                highest = max(highest, float(vals.max()))
                errors = np.array([[median-vals.min()], [vals.max()-median]])
                ax.bar(pos, median, width=.31, color=COLORS[name], zorder=3,
                       yerr=errors if len(vals)>1 else None, capsize=3,
                       error_kw={"elinewidth": 1.1, "ecolor": "#243647"})
        ax.set_title(f"N = {n:,}\n{n*4/2**20:.3g} MiB input", fontsize=13, pad=12)
        ax.set_xticks(x, [str(c) for c in caps])
        ax.tick_params(axis="x", labelrotation=35 if len(caps)>5 else 0)
        ax.set_xlabel("Maximum blocks")
        ax.set_ylabel("Reduction time (µs)")
        ax.set_ylim(0, highest*1.18 if highest>0 else 1)
        ax.grid(axis="y", color="#E5EAF0", zorder=0)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(sizes):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color=COLORS[v], label=v.capitalize()) for v in VERSIONS],
               loc="upper center", ncol=2, frameon=False)
    fig.text(.5, .015, "Bars: median batch time (or reported time).\n"
             "Whiskers: min–max batches, not confidence intervals.\n"
             "Independent y axes. Missing data: N/A.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .65/(4*nrows+1), 1, .92 if nrows==1 else .96), h_pad=2.4)
    for ext in ("png", "svg"):
        fig.savefig(output/f"timing_comparison.{ext}", dpi=200)
    plt.close(fig)

    ratio = np.full((len(sizes), len(caps)), np.nan)
    for i, n in enumerate(sizes):
        for j, cap in enumerate(caps):
            row = lookup.get((n, cap), {})
            if all(v in row for v in VERSIONS):
                ratio[i,j] = float(np.median(row["stride"]))/float(np.median(row["shuffle"]))
    valid = ratio[np.isfinite(ratio)]
    span = max(.05, float(np.max(np.abs(valid-1)))*1.08) if valid.size else .05
    cmap = LinearSegmentedColormap.from_list("ratio", ["#DDE6F1", "#FFFFFF", "#B9E2D9"])
    cmap.set_bad("#EEEEEE")
    fig, ax = plt.subplots(figsize=(max(7, len(caps)*1.55+2), max(4, len(sizes)*.8+1.8)))
    ax.imshow(np.ma.masked_invalid(ratio), cmap=cmap,
              norm=TwoSlopeNorm(vmin=1-span, vcenter=1, vmax=1+span), aspect="auto")
    ax.set_xticks(range(len(caps)), [str(c) for c in caps])
    ax.set_yticks(range(len(sizes)), [f"N = {n:,}" for n in sizes])
    ax.set_xlabel("Maximum blocks", labelpad=10)
    ax.tick_params(length=0, pad=10)
    for i in range(len(sizes)):
        for j in range(len(caps)):
            label = f"{ratio[i,j]:.3f}×" if np.isfinite(ratio[i,j]) else "N/A"
            ax.text(j, i, label, ha="center", va="center", fontsize=18,
                    color="#172D40", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Speedup = stride time / shuffle time", fontsize=14, pad=18)
    fig.text(.5, .02, "Above 1×: shuffle is faster. Below 1×: stride is faster.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(output/f"shuffle_speedup.{ext}", dpi=200)
    plt.close(fig)
    print(f"Charts saved to: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Default: reduction_results.txt in current/script directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--sizes", type=int, nargs="+", help="Default: all timed sizes in log")
    parser.add_argument("--blocks", type=int, nargs="+", help="Default: inferred from selected data")
    args = parser.parse_args()
    source = args.input
    if source is None:
        source = Path("reduction_results.txt")
        if not source.is_file():
            source = Path(__file__).resolve().parent/"reduction_results.txt"
    try:
        print(f"Reading: {source.resolve()}")
        plot(parse_log(source), args.out, args.sizes, args.blocks)
    except (OSError, ValueError, UnicodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

