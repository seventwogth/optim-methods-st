import math

A, B = -0.8, 0.8
WIDTH, HEIGHT, PAD = 1000, 600, 60


def f(x: float) -> float:
    return math.sin(5 * x) * math.cos(0.5 * x)


def to_screen(x: float, y: float, ymin: float, ymax: float) -> tuple[float, float]:
    sx = PAD + (x - A) * (WIDTH - 2 * PAD) / (B - A)
    sy = HEIGHT - PAD - (y - ymin) * (HEIGHT - 2 * PAD) / (ymax - ymin)
    return sx, sy


def find_extrema() -> tuple[tuple[float, float], tuple[float, float]]:
    n = 20000
    step = (B - A) / n
    x_min = x_max = A
    y_min = y_max = f(A)
    for i in range(1, n + 1):
        x = A + i * step
        y = f(x)
        if y < y_min:
            x_min, y_min = x, y
        if y > y_max:
            x_max, y_max = x, y
    return (x_min, y_min), (x_max, y_max)


def build_svg(path: str = "second/extrema_plot.svg") -> None:
    n = 1200
    pts = []
    ymin, ymax = float("inf"), float("-inf")
    for i in range(n + 1):
        x = A + (B - A) * i / n
        y = f(x)
        pts.append((x, y))
        ymin = min(ymin, y)
        ymax = max(ymax, y)

    margin = 0.15 * (ymax - ymin)
    ymin -= margin
    ymax += margin

    (x_min, y_min), (x_max, y_max) = find_extrema()
    polyline = " ".join(f"{to_screen(x, y, ymin, ymax)[0]:.2f},{to_screen(x, y, ymin, ymax)[1]:.2f}" for x, y in pts)

    x_axis_y = to_screen(0, 0, ymin, ymax)[1]
    y_axis_x = to_screen(0, 0, ymin, ymax)[0]
    sx_min, sy_min = to_screen(x_min, y_min, ymin, ymax)
    sx_max, sy_max = to_screen(x_max, y_max, ymin, ymax)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{PAD}" y1="{x_axis_y:.2f}" x2="{WIDTH-PAD}" y2="{x_axis_y:.2f}" stroke="#888"/>
<line x1="{y_axis_x:.2f}" y1="{PAD}" x2="{y_axis_x:.2f}" y2="{HEIGHT-PAD}" stroke="#888"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline}"/>
<circle cx="{sx_max:.2f}" cy="{sy_max:.2f}" r="6" fill="#2ca02c"/>
<text x="{sx_max+10:.2f}" y="{sy_max-10:.2f}" font-size="18" fill="#2ca02c">max ({x_max:.4f}, {y_max:.4f})</text>
<circle cx="{sx_min:.2f}" cy="{sy_min:.2f}" r="6" fill="#d62728"/>
<text x="{sx_min+10:.2f}" y="{sy_min+24:.2f}" font-size="18" fill="#d62728">min ({x_min:.4f}, {y_min:.4f})</text>
<text x="{PAD}" y="30" font-size="22">f(x)=sin(5x)*cos(0.5x), интервал [-0.8, 0.8]</text>
</svg>
'''

    with open(path, "w", encoding="utf-8") as out:
        out.write(svg)
    print(f"saved: {path}")


if __name__ == "__main__":
    build_svg()
