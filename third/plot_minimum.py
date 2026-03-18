import matplotlib.pyplot as plt

from quadratic_approximation import A, B, f, quadratic_min

OUT = 'third/minimum.png'


def main() -> None:
    xs = [A + (B - A) * i / 1000 for i in range(1001)]
    ys = [f(x) for x in xs]
    x_min, y_min, _ = quadratic_min()

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label='f(x) = x^2 - 3x + x ln x')
    plt.scatter([x_min], [y_min], color='red', zorder=3, label=f'min ({x_min:.4f}, {y_min:.4f})')
    plt.title('Метод квадратичной аппроксимации, вариант 16')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()
