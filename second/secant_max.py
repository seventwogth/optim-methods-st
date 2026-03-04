import math

A, B = -4.0, 4.0
EPS = 1e-8


def f(x: float) -> float:
    return math.sin(5 * x) * math.cos(0.5 * x)


def df(x: float) -> float:
    return 5 * math.cos(5 * x) * math.cos(0.5 * x) - 0.5 * math.sin(5 * x) * math.sin(0.5 * x)


def bracket_stationary_max() -> tuple[float, float]:
    n = 20000
    step = (B - A) / n
    x_best, y_best = A, f(A)
    for i in range(1, n + 1):
        x = A + i * step
        y = f(x)
        if y > y_best:
            x_best, y_best = x, y
    l, r = max(A, x_best - 0.4), min(B, x_best + 0.4)
    return l, r


def secant_root(a: float, b: float, eps: float = EPS) -> tuple[float, float, int]:
    x0, x1 = a, b
    f0, f1 = df(x0), df(x1)
    for it in range(1, 201):
        den = f1 - f0
        if abs(den) < 1e-14:
            break
        x2 = x1 - f1 * (x1 - x0) / den
        if abs(x2 - x1) <= eps and abs(df(x2)) <= eps:
            return x2, f(x2), it
        x0, x1 = x1, x2
        f0, f1 = f1, df(x1)
    return x1, f(x1), it


l, r = bracket_stationary_max()
x, y, iters = secant_root(l, r)
print(f"x_max = {x:.12f}")
print(f"f(x_max) = {y:.12f}")
print(f"iterations = {iters}")
