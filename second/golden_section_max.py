import math

A, B = -4.0, 4.0
EPS = 1e-8


def f(x: float) -> float:
    return math.sin(5 * x) * math.cos(0.5 * x)


def coarse_max_interval(a: float = A, b: float = B, n: int = 20000) -> tuple[float, float]:
    step = (b - a) / n
    x_best, y_best = a, f(a)
    for i in range(1, n + 1):
        x = a + i * step
        y = f(x)
        if y > y_best:
            x_best, y_best = x, y
    return max(a, x_best - 0.4), min(b, x_best + 0.4)


def golden_section_max(a: float, b: float, eps: float = EPS) -> tuple[float, float, int]:
    tau = (math.sqrt(5) - 1) / 2
    x1, x2 = b - tau * (b - a), a + tau * (b - a)
    f1, f2 = f(x1), f(x2)
    it = 0
    while (b - a) / 2 > eps:
        if f1 >= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - tau * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
        it += 1
    x = (a + b) / 2
    return x, f(x), it


l, r = coarse_max_interval()
x, y, iters = golden_section_max(l, r)
print(f"x_max = {x:.12f}")
print(f"f(x_max) = {y:.12f}")
print(f"iterations = {iters}")
