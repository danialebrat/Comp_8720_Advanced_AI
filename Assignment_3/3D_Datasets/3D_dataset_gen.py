#%%
import matplotlib.pyplot as plt
from matplotlib import ticker

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn import manifold, datasets

n_samples = 1500
S_points, S_color = datasets.make_s_curve(n_samples, random_state=0)
Swiss_points, Swiss_colors = datasets.make_swiss_roll(n_samples=n_samples, random_state=0)
#%%
def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


plot_3d(S_points, S_color, "Original S-curve samples")
plot_3d(Swiss_points, Swiss_colors, "Original S-curve samples")

#%%
#Manifold Learning
n_neighbors = 12  # neighborhood which is used to recover the locally linear structure
n_components = 2  # number of coordinates for the manifold


params = {
    "n_neighbors": n_neighbors,
    "n_components": n_components,
    "eigen_solver": "auto",
    "random_state": 0,
}

def lle_standard(input_points, params):
    lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_standard.fit_transform(input_points)

def lle_ltsa(input_points, params):
    lle_ltsad = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_ltsa.fit_transform(input_points)

def lle_hessian(input_points, params):
    lle_hessian = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_hessian.fit_transform(input_points)

def lle_mod(input_points, params):
    lle_mod = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_mod.fit_transform(input_points)


S_standard = lle_standard(S_points)
Swiss_standard = lle_standard(Swiss_points)

S_ltsa = lle_ltsa(S_points)
Swiss_ltsa = lle_ltsa(Swiss_points)

S_hessian = lle_hessian(S_points)
Swiss_hessian = lle_hessian(Swiss_points)

S_mod = lle_mod(S_points)
Swiss_mod = lle_mod(Swiss_points)


fig, axs = plt.subplots(
    nrows=2, ncols=2, figsize=(7, 7), facecolor="white", constrained_layout=True
)
fig.suptitle("Locally Linear Embeddings", size=16)

lle_methods = [
    ("Standard locally linear embedding", S_standard),
    ("Local tangent space alignment", S_ltsa),
    ("Hessian eigenmap", S_hessian),
    ("Modified locally linear embedding", S_mod),
]
for ax, method in zip(axs.flat, lle_methods):
    name, points = method
    add_2d_scatter(ax, points, S_color, name)

plt.show()
# %%
