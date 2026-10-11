"""Edit the three times below (milliseconds), then run this script.
Dependency: pip install matplotlib
"""
from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Change only these three values. All times must be in milliseconds.
CPU_MS = 14652.7
NAIVE_GPU_MS = 33.218
SHARED_GPU_MS = 27.912


def plot_times(cpu_ms, naive_ms, shared_ms):
    times = [cpu_ms, naive_ms, shared_ms]
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("All execution times must be finite and greater than zero.")

    names = ["CPU", "Naive GPU", "Shared GPU"]
    colors = ["#94A3B8", "#2563EB", "#14B8A6"]
    speedup = cpu_ms / naive_ms
    shared_speedup = naive_ms / shared_ms
    change_pct = (shared_ms / naive_ms - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.29,
                        top=0.76, wspace=0.34)
    fig.suptitle("CUDA execution time", x=0.055, y=0.955,
                 ha="left", fontsize=24, fontweight="bold")
    fig.text(0.055, 0.89,
             "Lower is better | Panels use independent linear scales",
             fontsize=12, color="#64748B")

    panels = [(names, times, colors, "All implementations"),
              (names[1:], times[1:], colors[1:], "GPU detail")]
    for ax, (labels, values, palette, title) in zip(axes, panels):
        bars = ax.bar(labels, values, color=palette, width=0.55, zorder=3)
        ax.set_title(title, loc="left", fontsize=15, pad=18,
                     fontweight="bold")
        ax.set_ylabel("Execution time (ms)", fontsize=12)
        # Start at zero and reserve space above the largest bar for labels.
        ax.set_ylim(0, max(values) * 1.25)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 5),
                            useOffset=False)
        ax.bar_label(bars, labels=[f"{t:.6g} ms" for t in values],
                      padding=7, fontsize=12, fontweight="bold")
        ax.grid(axis="y", color="#E2E8F0")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=11, pad=8)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color("#CBD5E1")

    fig.text(0.085, 0.18, f"{speedup:.4g}×  Naive GPU vs CPU",
             fontsize=16, fontweight="bold", color=colors[1])
    fig.text(0.085, 0.13, "Speedup = CPU time / naive GPU time",
             fontsize=11, color="#64748B")
    fig.text(0.59, 0.18, f"{shared_speedup:.4g}×  Shared vs naive GPU",
             fontsize=16, fontweight="bold", color="#0F766E")
    if abs(change_pct) < 0.05:
        comparison = "Shared and naive times are approximately equal."
    else:
        direction = "longer" if change_pct > 0 else "less time"
        comparison = f"Shared takes {abs(change_pct):.3g}% {direction}."
    fig.text(0.59, 0.13, comparison, fontsize=11, color="#64748B")
    fig.text(0.055, 0.045,
             "Speedup = baseline time / measured time. "
             "Above 1× is faster; below 1× is slower.",
             fontsize=10, color="#64748B")
    return fig


if __name__ == "__main__":
    figure = plot_times(CPU_MS, NAIVE_GPU_MS, SHARED_GPU_MS)
    output_dir = Path(__file__).resolve().parent
    for extension in ("png", "svg"):
        output = output_dir / f"cuda_times.{extension}"
        figure.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output}")
    plt.show()
