import math

A, B = -4.0, 4.0
EPS = 1e-8


def f(x: float) -> float:
    return math.sin(5 * x) * math.cos(0.5 * x)


def df(x: float) -> float:
    return 5 * math.cos(5 * x) * math.cos(0.5 * x) - 0.5 * math.sin(5 * x) * math.sin(0.5 * x)


def d2f(x: float) -> float:
    return -25.25 * math.sin(5 * x) * math.cos(0.5 * x) - 5 * math.cos(5 * x) * math.sin(0.5 * x)


def initial_max_guess() -> float:
    n = 20000
    step = (B - A) / n
    x_best, y_best = A, f(A)
    for i in range(1, n + 1):
        x = A + i * step
        y = f(x)
        if y > y_best:
            x_best, y_best = x, y
    return x_best


def newton_stationary(x0: float, eps: float = EPS) -> tuple[float, float, int]:
    x = x0
    for it in range(1, 101):
        x_next = x - df(x) / d2f(x)
        if abs(x_next - x) <= eps and abs(df(x_next)) <= eps:
            return x_next, f(x_next), it
        x = x_next
    return x, f(x), it


x0 = initial_max_guess()
x, y, iters = newton_stationary(x0)
print(f"x_max = {x:.12f}")
print(f"f(x_max) = {y:.12f}")
print(f"iterations = {iters}")
