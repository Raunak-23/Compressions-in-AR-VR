"""Script to generate visual flowcharts and diagrams for the XR Compression Lab PDF guide."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path("experiments")
OUT_DIR.mkdir(exist_ok=True)

# Set common style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def create_box(ax, xy, width, height, text, subtext="", box_color="#2563EB", text_color="#FFFFFF", edge_color=None):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=box_color,
        edgecolor=edge_color or box_color,
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(rect)
    cx = xy[0] + width / 2.0
    if subtext:
        cy1 = xy[1] + height * 0.62
        cy2 = xy[1] + height * 0.30
        ax.text(cx, cy1, text, color=text_color, fontsize=8.5, weight='bold', ha='center', va='center', zorder=3)
        ax.text(cx, cy2, subtext, color=text_color, fontsize=7.0, ha='center', va='center', zorder=3, alpha=0.9)
    else:
        cy = xy[1] + height / 2.0
        ax.text(cx, cy, text, color=text_color, fontsize=8.5, weight='bold', ha='center', va='center', zorder=3)

def create_arrow(ax, start_xy, end_xy, label="", color="#475569"):
    ax.annotate(
        "", xy=end_xy, xytext=start_xy,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=11),
        zorder=1
    )
    if label:
        mx = (start_xy[0] + end_xy[0]) / 2.0
        my = (start_xy[1] + end_xy[1]) / 2.0 + 0.09
        ax.text(mx, my, label, color="#1E293B", fontsize=6.8, ha='center', va='bottom', weight='semibold', zorder=4)

# 1. Flow Diagram: Classical Lossless Encoding Pipeline
def gen_lossless_flow():
    fig, ax = plt.subplots(figsize=(10.5, 2.2), dpi=220)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    stages = [
        ("Raw Media Bytes", "RGB / PCM Stream", "#0F172A", "#FFFFFF"),
        ("Run-Length Coding", "Count-Value Pairs", "#2563EB", "#FFFFFF"),
        ("Symbol Frequencies", "Counter / Min-Heap", "#0284C7", "#FFFFFF"),
        ("Huffman Bit-Pack", "Prefix Tree Coding", "#0D9488", "#FFFFFF"),
        ("LZ77 Sliding Window", "Match Length & Offset", "#16A34A", "#FFFFFF"),
        ("Deflate / LZMA", "Compressed Bitstream", "#7C3AED", "#FFFFFF")
    ]

    bw = 1.35
    bh = 1.1
    spacing = 0.38
    start_x = 0.3
    y = 0.55

    for i, (title, sub, bg, fg) in enumerate(stages):
        x = start_x + i * (bw + spacing)
        create_box(ax, (x, y), bw, bh, title, sub, bg, fg)
        if i < len(stages) - 1:
            next_x = x + bw + spacing
            create_arrow(ax, (x + bw, y + bh / 2), (next_x, y + bh / 2))

    plt.title("Flow A: Classical Lossless Codec Pipeline (RLE, Huffman, zlib Deflate, LZMA)", fontsize=10, weight='bold', color="#0F172A", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "flow_lossless.png", bbox_inches='tight')
    plt.close()

# 2. Flow Diagram: Transform Lossy Video/Image Compression
def gen_lossy_flow():
    fig, ax = plt.subplots(figsize=(10.5, 2.2), dpi=220)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    stages = [
        ("RGB 1280x720", "Input Video Frame", "#0F172A", "#FFFFFF"),
        ("4:2:0 Chroma Subsample", "Halve Cb/Cr Res", "#2563EB", "#FFFFFF"),
        ("8x8 Block DCT-II", "Spatial to Frequency", "#0284C7", "#FFFFFF"),
        ("Annex-K Quantization", "Matrix * Q-Ladder", "#D97706", "#FFFFFF"),
        ("Zig-Zag & Entropy", "AC Zeros + DC DPCM", "#16A34A", "#FFFFFF"),
        ("H.264 / JPEG Output", "IDR / NAL Packets", "#7C3AED", "#FFFFFF")
    ]

    bw = 1.35
    bh = 1.1
    spacing = 0.38
    start_x = 0.3
    y = 0.55

    for i, (title, sub, bg, fg) in enumerate(stages):
        x = start_x + i * (bw + spacing)
        create_box(ax, (x, y), bw, bh, title, sub, bg, fg)
        if i < len(stages) - 1:
            next_x = x + bw + spacing
            create_arrow(ax, (x + bw, y + bh / 2), (next_x, y + bh / 2))

    plt.title("Flow B: Transform Lossy Coding Pipeline (Chroma Subsampling, DCT-II, Quantization, H.264)", fontsize=10, weight='bold', color="#0F172A", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "flow_lossy_transform.png", bbox_inches='tight')
    plt.close()

# 3. Flow Diagram: 4x4 Viewport-Aware Tiling Pipeline
def gen_viewport_flow():
    fig, ax = plt.subplots(figsize=(10.5, 2.2), dpi=220)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    stages = [
        ("Viewer Gaze", "Touch (nx,ny) / Yaw-Pitch", "#0F172A", "#FFFFFF"),
        ("Frustum RoI Box", "100° x 62° Intersection", "#2563EB", "#FFFFFF"),
        ("16-Tile Grid (4x4)", "Gapless Partitioning", "#0284C7", "#FFFFFF"),
        ("3-Tier Assignment", "High, Chebyshev Med, Low", "#D97706", "#FFFFFF"),
        ("Knapsack Allocator", "Budget Rate-Distortion", "#16A34A", "#FFFFFF"),
        ("Seamless Stitch", "Full Canvas & Seam MSE", "#7C3AED", "#FFFFFF")
    ]

    bw = 1.35
    bh = 1.1
    spacing = 0.38
    start_x = 0.3
    y = 0.55

    for i, (title, sub, bg, fg) in enumerate(stages):
        x = start_x + i * (bw + spacing)
        create_box(ax, (x, y), bw, bh, title, sub, bg, fg)
        if i < len(stages) - 1:
            next_x = x + bw + spacing
            create_arrow(ax, (x + bw, y + bh / 2), (next_x, y + bh / 2))

    plt.title("Flow C: 4x4 Viewport-Aware Tile Allocation & Reconstruction Flow", fontsize=10, weight='bold', color="#0F172A", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "flow_viewport_tiles.png", bbox_inches='tight')
    plt.close()

# 4. Flow Diagram: Spatial Audio Pipeline
def gen_audio_flow():
    fig, ax = plt.subplots(figsize=(10.5, 2.2), dpi=220)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    stages = [
        ("4ch FOA (B-format)", "W, X, Y, Z Field", "#0F172A", "#FFFFFF"),
        ("Profile Folding", "4ch -> Stereo -> Mono", "#2563EB", "#FFFFFF"),
        ("Scalar Quantize", "float32 -> int16 PCM", "#0284C7", "#FFFFFF"),
        ("Predictive DPCM", "Successive Sample Deltas", "#D97706", "#FFFFFF"),
        ("Psychoacoustic Opus", "24 - 128 kbps (SILK/MDCT)", "#16A34A", "#FFFFFF"),
        ("SRTP Audio Stream", "48 kHz Paced 20ms Pkts", "#7C3AED", "#FFFFFF")
    ]

    bw = 1.35
    bh = 1.1
    spacing = 0.38
    start_x = 0.3
    y = 0.55

    for i, (title, sub, bg, fg) in enumerate(stages):
        x = start_x + i * (bw + spacing)
        create_box(ax, (x, y), bw, bh, title, sub, bg, fg)
        if i < len(stages) - 1:
            next_x = x + bw + spacing
            create_arrow(ax, (x + bw, y + bh / 2), (next_x, y + bh / 2))

    plt.title("Flow D: Spatial Audio Compression, Ambisonic Folding & Opus Encoding Flow", fontsize=10, weight='bold', color="#0F172A", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "flow_spatial_audio.png", bbox_inches='tight')
    plt.close()

# 5. Flow Diagram: Closed-Loop Bandwidth Adaptation (ABR)
def gen_abr_flow():
    fig, ax = plt.subplots(figsize=(10.5, 2.2), dpi=220)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    stages = [
        ("WebRTC Receiver Stats", "bytesReceived / dT", "#0F172A", "#FFFFFF"),
        ("EWMA Estimator", "Smoothed kbps + Variance", "#2563EB", "#FFFFFF"),
        ("Ladder Decision", "Ultra / High / Med / Low", "#0284C7", "#FFFFFF"),
        ("5-Knob Adaptation", "Scale, FPS, Q, Blur, Audio", "#D97706", "#FFFFFF"),
        ("Paced Invariant Canvas", "Fixed 1280x720 Container", "#16A34A", "#FFFFFF"),
        ("P2P SRTP Delivery", "Low Jitter / Zero Renego", "#7C3AED", "#FFFFFF")
    ]

    bw = 1.35
    bh = 1.1
    spacing = 0.38
    start_x = 0.3
    y = 0.55

    for i, (title, sub, bg, fg) in enumerate(stages):
        x = start_x + i * (bw + spacing)
        create_box(ax, (x, y), bw, bh, title, sub, bg, fg)
        if i < len(stages) - 1:
            next_x = x + bw + spacing
            create_arrow(ax, (x + bw, y + bh / 2), (next_x, y + bh / 2))

    plt.title("Flow E: Closed-Loop Network Bandwidth Adaptation (EWMA -> Ladder -> Paced Canvas)", fontsize=10, weight='bold', color="#0F172A", pad=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "flow_adaptive_abr.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    gen_lossless_flow()
    gen_lossy_flow()
    gen_viewport_flow()
    gen_audio_flow()
    gen_abr_flow()
    print("All 5 flow diagrams generated successfully in experiments/")
