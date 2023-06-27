from pytest import fixture
import barnacle.fista as fista
import pytest
import numpy as np
from scipy.optimize import check_grad


@pytest.mark.parametrize(
    "x,prox_x,reg",
    (
        (4, 3, 1),
        (-4, -3, 1),
        (1, 0, 1),
        (-1, 0, 1),
        (1, 0.5, 0.5),
        (-1, -0.5, 0.5),
        (1, 0, 2),
        (-1, 0, 2),
        ([1, 2], [0, 0.5], 1.5),
        ([-1, -2], [0, -0.5], 1.5),
    ),
)
def test_l1_prox(x, prox_x, reg):
    x = np.array(x).reshape(1, -1)
    prox_x = np.array(prox_x).reshape(1, -1)
    np.testing.assert_allclose(prox_x, fista.l1_prox(x, reg))


@pytest.mark.parametrize(
    "x,prox_x",
    (
        ([4, 0], [1, 0]),
        ([-4, 0], [-1, 0]),
        ([0.5, 0], [0.5, 0]),
        ([-0.5, 0], [-0.5, 0]),
        ([1, 1], [1 / np.sqrt(2), 1 / np.sqrt(2)]),
        ([1, -1], [1 / np.sqrt(2), -1 / np.sqrt(2)]),
    ),
)
def test_l2ball_prox(x, prox_x):
    x = np.array(x).reshape(1, -1)
    prox_x = np.array(prox_x).reshape(1, -1)
    np.testing.assert_allclose(prox_x, fista.l2ball_prox(x, None))


@pytest.mark.parametrize(
    "x,prox_x",
    (
        ([4, 0], [4, 0]),
        ([-4, 1], [0, 1]),
        ([-1, -2], [0, 0]),
    ),
)
def test_nn_prox(x, prox_x):
    x = np.array(x).reshape(1, -1)
    prox_x = np.array(prox_x).reshape(1, -1)
    np.testing.assert_allclose(prox_x, fista.nn_prox(x, None))


@fixture
def seed():
    return 0


@fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("A_shape", ((2, 3), (5, 3), (4, 1), (1, 4)))
@pytest.mark.parametrize("rank", (1, 2, 5))
def test_create_loss_is_correct_for_vectors(rng, A_shape, rank):
    A = rng.standard_normal(A_shape)
    b = rng.standard_normal((A_shape[0], rank))
    x = rng.standard_normal((A_shape[1], rank))

    AtA = A.T @ A
    At_b = A.T @ b
    ssb = np.sum(b**2)
    sse = np.sum((A @ x - b) ** 2)

    compute_smooth_loss = fista.create_loss(AtA, At_b)
    assert pytest.approx(compute_smooth_loss(x)) == 0.5 * (sse - ssb)


@pytest.mark.parametrize("A_shape", ((2, 3), (5, 3), (4, 1), (1, 4)))
@pytest.mark.parametrize("rank", (1, 2, 5))
def test_create_loss_is_correct_for_optimal_point(rng, A_shape, rank):
    A = rng.standard_normal(A_shape)
    x = rng.standard_normal((A_shape[1], rank))

    b = A @ x
    AtA = A.T @ A
    At_b = A.T @ b
    ssb = np.sum(b**2)

    compute_new_smooth_loss = fista.create_loss(AtA, At_b)
    assert pytest.approx(compute_new_smooth_loss(x)) == -0.5 * ssb


@pytest.mark.parametrize("A_shape", ((2, 3), (5, 3), (4, 1), (1, 4)))
@pytest.mark.parametrize("rank", (1, 2, 5))
def test_gradient_is_correct_analytically(rng, A_shape, rank):
    A = rng.standard_normal(A_shape)
    b = rng.standard_normal((A_shape[0], rank))
    x = rng.standard_normal((A_shape[1], rank))

    AtA = A.T @ A
    At_b = A.T @ b

    compute_smooth_grad = fista.create_gradient(AtA, At_b)
    np.testing.assert_allclose(compute_smooth_grad(x), AtA @ x - At_b, atol=1e-10)


@pytest.mark.parametrize("A_shape", ((2, 3), (5, 3), (4, 1), (1, 4)))
@pytest.mark.parametrize("rank", (1, 2, 5))
def test_gradient_is_correct_numerically(rng, A_shape, rank):
    A = rng.standard_normal(A_shape)
    b = rng.standard_normal((A_shape[0], rank))
    x = rng.standard_normal((A_shape[1], rank))

    AtA = A.T @ A
    At_b = A.T @ b

    compute_smooth_grad = fista.create_gradient(AtA, At_b)
    compute_smooth_loss = fista.create_loss(AtA, At_b)
    assert check_grad(
        lambda z: compute_smooth_loss(z.reshape(x.shape)),
        lambda z: compute_smooth_grad(z.reshape(x.shape)).ravel(),
        x.ravel(),
    ) == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize("A_shape", ((2, 3), (5, 3), (4, 1), (1, 4)))
@pytest.mark.parametrize("rank", (1, 2, 5))
def test_gradient_is_correct_in_minimum(rng, A_shape, rank):
    A = rng.standard_normal(A_shape)
    x = rng.standard_normal((A_shape[1], rank))

    AtA = A.T @ A
    b = A @ x

    AtA = A.T @ A
    At_b = A.T @ b

    compute_smooth_grad = fista.create_gradient(AtA, At_b)
    np.testing.assert_allclose(compute_smooth_grad(x), 0, atol=1e-10)


@pytest.mark.parametrize("dim", (1, 2, 5))
@pytest.mark.parametrize("rank", (1, 2, 5))
@pytest.mark.parametrize("reg", [0, 0.001, 0.1, 0.5, 1, 100])
@pytest.mark.parametrize("line_search", [True, False])
def test_minimize_fista_is_prox_with_identity(rng, dim, rank, reg, line_search):
    I = np.identity(dim)
    b = rng.standard_normal((dim, rank))
    x = rng.standard_normal((dim, rank))

    y = fista.minimise_fista(
        lhs=I,
        rhs=b,
        init=np.zeros_like(x),
        l1_reg=reg,
        prox=fista.l1_prox,
        n_iter=1000,
        tol=1e-8,
        line_search=line_search,
    )

    prox = fista.l1_prox(b, reg)
    np.testing.assert_allclose(prox, y)
