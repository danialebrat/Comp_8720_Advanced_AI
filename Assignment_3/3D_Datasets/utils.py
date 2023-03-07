import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import manifold


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


# Manifolds clusterings

def lle_standard(input_points, params):
    lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_standard.fit_transform(input_points)


def lle_ltsa(input_points, params):
    lle_ltsad = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_ltsad.fit_transform(input_points)


def lle_hessian(input_points, params):
    lle_hessian = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_hessian.fit_transform(input_points)


def lle_mod(input_points, params):
    lle_mod = manifold.LocallyLinearEmbedding(method="standard", **params)
    return lle_mod.fit_transform(input_points)


def isomap(input_points, num_components, num_neighbors, p=1):
    isomap = manifold.Isomap(n_neighbors=num_neighbors, 
                             n_components=num_components,
                             p=1)
    return isomap.fit_transform(input_points)


def spectral_Laplacian_Eigen_map(input_points, num_components, num_neighbors, p=1):
    spectral = manifold.SpectralEmbedding(n_neighbors=num_neighbors, 
                                          n_components=num_components)
    return spectral.fit_transform(input_points)