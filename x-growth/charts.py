"""Render the data charts that go with queued posts. Output: media/*.png (1600x900)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "media"
SURFACE, INK, INK2, MUTED, BLUE, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#2a78d6", "#b9b8b2"
plt.rcParams.update({"font.family": "Inter", "font.size": 15, "text.color": INK})


def frame(title, subtitle, source):
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    fig.text(0.06, 0.90, title, fontsize=34, fontweight="bold", ha="left")
    fig.text(0.06, 0.845, subtitle, fontsize=18, color=INK2, ha="left")
    fig.text(0.06, 0.04, source, fontsize=13, color=MUTED, ha="left")
    fig.text(0.94, 0.04, "@aleromlan", fontsize=13, color=MUTED, ha="right")
    ax = fig.add_axes([0.06, 0.13, 0.88, 0.64], facecolor=SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK2, labelsize=18)
    ax.set_yticks([])
    return fig, ax


def bars(ax, labels, values, colors, texts, ymax):
    w = 0.62
    for i, (v, c, t) in enumerate(zip(values, colors, texts)):
        ax.bar(i, v, width=w, color=c, linewidth=0)
        ax.text(i, v + ymax * 0.025, t, ha="center", va="bottom", fontsize=26, fontweight="bold")
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xticks(range(len(labels)), labels)
    ax.axhline(0, color=GRAY, linewidth=1.2)


fig, ax = frame("Anthropic revenue, from the S-1",
                "Half a year of 2026 is 3.5x all of 2025",
                "Source: Anthropic S-1 filing, Oct 2026. 2023–2025 are full years; 2026 is January–June only.")
bars(ax, ["2023", "2024", "2025", "2026 (Jan–Jun)"], [0.041, 0.386, 4.59, 16.29],
     [BLUE] * 4, ["$41M", "$386M", "$4.6B", "$16.3B"], 16.29)
fig.savefig(OUT / "003-anthropic-s1.png", facecolor=SURFACE)

fig, ax = frame("Hard diagnostic cases solved correctly",
                "304 NEJM case reports. Doctors had textbooks and internet access.",
                "Source: Microsoft AI, Science, Oct 22 2026. Text-only cases, generalist physicians, not live patients.")
bars(ax, ["Microsoft MAI-DxO", "Generalist doctors"], [83.2, 36.1], [BLUE, GRAY], ["83%", "36%"], 100)
fig.savefig(OUT / "001-microsoft-diagnosis.png", facecolor=SURFACE)
