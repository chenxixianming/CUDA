import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

C = A @ B

GREEN = "#087D64"
BLUE = "#226CC0"
ORANGE = "#A65B00"
DARK = "#142F40"
GRAY = "#778793"

fig, ax = plt.subplots(figsize=(16, 7.4))
ax.set(xlim=(0, 16), ylim=(1.6, 9))
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def text(x, y, value, size=26, color=DARK, bold=False):
    ax.text(
        x, y, value,
        ha="center", va="center",
        fontsize=size, color=color,
        fontweight="bold" if bold else "normal"
    )


def draw_matrix(matrix, x, name, mode, color, background, label):
    # Highlight a row, column, or output element.
    regions = {
        "row":  (x - 1.32, 5.10, 2.64, 0.80),
        "col":  (x - 0.40, 4.24, 0.80, 2.52),
        "cell": (x - 0.40, 5.10, 0.80, 0.80),
    }

    left, bottom, width, height = regions[mode]
    ax.add_patch(FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        facecolor=background, edgecolor=color, linewidth=2
    ))

    for row in range(3):
        for col in range(3):
            selected = (
                (mode == "row" and row == 1)
                or (mode == "col" and col == 1)
                or (mode == "cell" and row == col == 1)
            )

            text(
                x + (col - 1) * 0.88,
                6.36 - row * 0.86,
                str(matrix[row, col]),
                size=29,
                color=color if selected else GRAY,
                bold=selected
            )

    # Matrix brackets.
    for side in (-1, 1):
        bx = x + side * 1.48
        tip = bx - side * 0.16
        ax.plot(
            [tip, bx, bx, tip],
            [6.88, 6.88, 4.13, 4.13],
            color=DARK, linewidth=2
        )

    text(x, 7.33, name, size=28)
    text(x, 3.77, label, size=20, color=color)


# text(8, 8.35, "Row × Column = One Element", size=30, bold=True)

draw_matrix(A, 3.1, "$A$", "row",
            GREEN, "#E6F5EE", "Row 2")

draw_matrix(B, 8.0, "$B$", "col",
            BLUE, "#E8F1FF", "Column 2")

draw_matrix(C, 12.9, "$C = AB$", "cell",
            ORANGE, "#FFF1D8", r"Element $C_{2,2}$")

text(5.55, 5.50, "x", size=38)
text(10.45, 5.50, "=", size=38)

# Expand the dot product.
text(
    8, 2.55,
    r"$C_{i,j} = \sum_{k=1}^{N} A_{i,k}\,B_{k,j}$",
    size=32
)

plt.savefig("matrix_multiplication.png", dpi=180, facecolor="white")
plt.show()