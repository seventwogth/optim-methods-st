import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [0.88, 0.89, 0.01],
    [2.01, 2.06, 1.45],
    [2.99, 2.83, 4.46],
    [4.01, 4.08, 1.04],
    [5.27, 4.96, -0.00]
])

X = data[:, :2]
Z = data[:, 2]

def kmeans_2(X, max_iter=100, eps=1e-9):
    centers = np.array([X[1], X[3]], dtype=float)

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(2)
        ])

        if np.linalg.norm(new_centers - centers) < eps:
            break

        centers = new_centers

    return centers

centers = kmeans_2(X)

def unpack(params):
    c = params[:4].reshape(2, 2)
    sigma = np.exp(params[4:6])
    w = params[6:8]
    b = params[8]
    return c, sigma, w, b

def predict(params, X):
    c, sigma, w, b = unpack(params)
    r2 = ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    phi = np.exp(-r2 / (2 * sigma[None, :] ** 2))
    return b + phi @ w

def loss_and_grad(params):
    c, sigma, w, b = unpack(params)

    r = X[:, None, :] - c[None, :, :]
    r2 = (r ** 2).sum(axis=2)
    phi = np.exp(-r2 / (2 * sigma[None, :] ** 2))

    pred = b + phi @ w
    e = pred - Z

    loss = 0.5 * np.mean(e ** 2)

    dc = np.zeros_like(c)
    dsigma = np.zeros_like(sigma)
    dw = np.zeros_like(w)
    db = np.mean(e)

    for j in range(2):
        dw[j] = np.mean(e * phi[:, j])
        dc[j] = np.mean(
            (e * w[j] * phi[:, j])[:, None] * r[:, j, :] / sigma[j] ** 2,
            axis=0
        )
        dsigma[j] = np.mean(
            e * w[j] * phi[:, j] * r2[:, j] / sigma[j] ** 3
        )

    dlog_sigma = dsigma * sigma

    grad = np.r_[dc.ravel(), dlog_sigma, dw, db]
    return loss, grad

dist = np.linalg.norm(centers[0] - centers[1])
sigma0 = np.array([dist / 2, dist / 2])

r2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
phi0 = np.exp(-r2 / (2 * sigma0[None, :] ** 2))

A = np.column_stack([phi0, np.ones(len(X))])
w1, w2, b = np.linalg.lstsq(A, Z, rcond=None)[0]

params = np.r_[centers.ravel(), np.log(sigma0), w1, w2, b]

loss_history = []

lr = 0.02
m = np.zeros_like(params)
v = np.zeros_like(params)

for t in range(1, 20001):
    loss, grad = loss_and_grad(params)
    loss_history.append(loss)

    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad ** 2

    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)

    params -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    params[4:6] = np.clip(params[4:6], np.log(0.05), np.log(20.0))

c, sigma, w, b = unpack(params)
pred = predict(params, X)
residuals = Z - pred

print("Параметры итоговой RBF-сети:")
print("centers =", c)
print("sigma =", sigma)
print("weights =", w)
print("bias =", b)
print("MSE =", 2 * loss_history[-1])
print("Loss = 0.5 * MSE =", loss_history[-1])

print("\nНевязки:")
for i in range(len(X)):
    print(
        f"{i}: z={Z[i]:.4f}, pred={pred[i]:.4f}, residual={residuals[i]:.4f}"
    )

print("\nАналитический вид модели:")
print(
    f"z(x,y) = {b:.4f}"
    f" + {w[0]:.4f} * exp(-((x - {c[0,0]:.4f})^2 + (y - {c[0,1]:.4f})^2) / (2 * {sigma[0]:.4f}^2))"
    f" + {w[1]:.4f} * exp(-((x - {c[1,0]:.4f})^2 + (y - {c[1,1]:.4f})^2) / (2 * {sigma[1]:.4f}^2))"
)

plt.figure()
plt.plot(loss_history)
plt.xlabel("Итерация")
plt.ylabel("Loss = 0.5 * MSE")
plt.title("Кривая обучения RBF-сети")
plt.grid(True)
plt.show()

x_grid = np.linspace(0, 6, 100)
y_grid = np.linspace(0, 6, 100)
Xg, Yg = np.meshgrid(x_grid, y_grid)
grid = np.column_stack([Xg.ravel(), Yg.ravel()])
Zg = predict(params, grid).reshape(Xg.shape)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.8)
ax1.scatter(X[:, 0], X[:, 1], Z, c="red", s=60)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_title("RBF-сеть: поверхность")

ax2 = fig.add_subplot(122)
contour = ax2.contourf(Xg, Yg, Zg, levels=25, cmap="viridis")
ax2.scatter(X[:, 0], X[:, 1], c=Z, edgecolors="black", s=80)
plt.colorbar(contour, ax=ax2)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("RBF-сеть: линии уровня")

plt.show()
