"""Draw a convolution diagram. Requires: numpy, matplotlib.
Run: python draw_convolution.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

X = np.arange(1, 26).reshape(5, 5)
W = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]])
K = W.shape[0]
R = K // 2
padded = np.pad(X, R, mode='constant')
Y = np.zeros_like(X)

# Same indexing as the CUDA example: stride 1, zero padding, no kernel flip.
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Y[i, j] = np.sum(padded[i:i+K, j:j+K] * W)

assert Y[2, 2] == 65
GREEN, BLUE, ORANGE = '#087D64', '#226CC0', '#A65B00'
DARK, GRAY = '#142F40', '#778793'
fig, ax = plt.subplots(figsize=(16, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 16), ylim=(0, 8))
ax.axis('off')


def text(x, y, value, size=24, color=DARK, bold=False):
    ax.text(x, y, value, ha='center', va='center', fontsize=size,
            color=color, fontweight='bold' if bold else 'normal')


def matrix(values, cx, selected, color, fill, name, caption):
    rows, cols = values.shape
    cell = 0.72
    cy = 4.6
    left = cx - cols * cell / 2
    top = cy + rows * cell / 2
    r0, c0, height, width = selected
    ax.add_patch(FancyBboxPatch(
        (left+c0*cell+0.03, top-(r0+height)*cell+0.03),
        width*cell-0.06, height*cell-0.06,
        boxstyle='round,pad=0.015,rounding_size=0.08',
        facecolor=fill, edgecolor=color, linewidth=2
    ))
    for r in range(rows):
        for c in range(cols):
            active = r0 <= r < r0+height and c0 <= c < c0+width
            text(left+(c+0.5)*cell, top-(r+0.5)*cell,
                 str(values[r, c]), size=25,
                 color=color if active else GRAY, bold=active)
    bottom = top - rows*cell
    for bx, step in [(left-0.12, 0.14), (left+cols*cell+0.12, -0.14)]:
        ax.plot([bx+step, bx, bx, bx+step],
                [top+0.08, top+0.08, bottom-0.08, bottom-0.08],
                color=DARK, linewidth=2)
    text(cx, 6.85, name, size=28)
    text(cx, 2.30, caption, size=20, color=color)


# text(8, 7.65, '2D Convolution: Multiply and Sum', size=30, bold=True)
matrix(X, 2.75, (1, 1, 3, 3), GREEN, '#E6F5EE', '$X$', 'Input patch')
matrix(W, 8.00, (0, 0, 3, 3), BLUE, '#E8F1FF', '$W$', '3 × 3 kernel')
matrix(Y, 13.25, (2, 2, 1, 1), ORANGE, '#FFF1D8', '$Y$', 'Output element')
text(5.55, 4.60, r'$\ast$', size=40)
text(10.45, 4.60, '=', size=34)

text(8, 1.25,
     r'$Y_{i,j}=\sum_{u=0}^{K-1}\quad\sum_{v=0}^{K-1}'
     r'X_{i+u-R,\,j+v-R}\,W_{u,v}$', size=29)
text(8, 0.37, r'$R=(K-1)/2$'+'     Stride 1 · Zero padding',
     size=15, color=GRAY)

output = Path(__file__).with_name('convolution.png')
fig.savefig(output, dpi=180, facecolor='white')
plt.close(fig)
print(output)
