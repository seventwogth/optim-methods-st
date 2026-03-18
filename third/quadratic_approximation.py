import math

A = 1.0
B = 2.0
EPS = 1e-4


def f(x: float) -> float:
    return x * x - 3 * x + x * math.log(x)


def vertex(x1: float, x2: float, x3: float) -> float:
    f1, f2, f3 = f(x1), f(x2), f(x3)
    a1 = x2 - x1
    a2 = x2 - x3
    num = a1 * a1 * (f2 - f3) - a2 * a2 * (f2 - f1)
    den = 2 * (a1 * (f2 - f3) - a2 * (f2 - f1))
    return x2 - num / den


def pick_triplet(points: list[float]) -> tuple[int, tuple[float, float, float]]:
    i = min(range(len(points)), key=lambda k: f(points[k]))
    if i == 0:
        return i, (points[0], points[1], points[2])
    if i == len(points) - 1:
        return i, (points[-3], points[-2], points[-1])
    return i, (points[i - 1], points[i], points[i + 1])


def quadratic_min(a: float = A, b: float = B, eps: float = EPS) -> tuple[float, float, list[float]]:
    x0 = (a + b) / 2
    d = (b - a) / 4
    x1, x2, x3 = x0 - d, x0, x0 + d
    seq = [x2]

    while True:
        prev = x2
        u = min(b, max(a, vertex(x1, x2, x3)))
        pts = sorted({a, b, x1, x2, x3, u})
        best, triple = pick_triplet(pts)
        if best == 0 or best == len(pts) - 1:
            seq.append(pts[best])
            return pts[best], f(pts[best]), seq
        x1, x2, x3 = triple
        seq.append(x2)
        if abs(x2 - prev) < eps:
            return x2, f(x2), seq


if __name__ == '__main__':
    x_min, y_min, seq = quadratic_min()
    for i, x in enumerate(seq):
        print(f'x[{i}] = {x:.6f}')
    print(f'\nx_min = {x_min:.6f}')
    print(f'f(x_min) = {y_min:.6f}')
    print(f'iterations = {len(seq) - 1}')
