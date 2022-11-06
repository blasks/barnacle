import numpy as np
import scipy.linalg as sla


def l1_prox(x, reg):
    sign = np.sign(x)
    return sign * np.maximum(0, np.abs(x) - reg)


def nn_l1_prox(x, reg):
    return np.maximum(0, x - reg)


def nn_l2ball_prox(x, reg):
    return l2ball_prox(nn_prox(x, None), None)


def nn_prox(x, reg):
    return np.maximum(0, x)


def l2ball_prox(x, reg):
    return x / np.maximum(1, np.linalg.norm(x, axis=0))


def _should_continue_backtracking(new_x, y, loss_new_x, loss_y, smooth_grad_y, lipschitz):
    # Based on FISTA with backtracking. Reformulation of this criterion:
    # F(new_optimal_x) > Q(new_optimal_x, old_momentum_x)
    # f(new_optimal_x) + g(new_optimal_x) > Q(new_optimal_x, old_momentum_x)
    # f(new_optimal_x) > Q(new_optimal_x, old_momentum_x) - g(new_optimal_x)
    # Combine with eq. 2.5 in Beck & Teboulle (2009) to obtain following condition
    # Modified slightly, increasing the threshold for the Lipschitz
    update_vector = new_x - y

    update_distance = np.sum(update_vector ** 2) * lipschitz / 2.5
    linearised_improvement = smooth_grad_y.ravel().T @ update_vector.ravel()

    return loss_new_x - loss_y > update_distance + linearised_improvement


def create_loss(AtA, At_b):
    def loss(x):
        iprod = np.sum(At_b * x)
        cp_norm = np.sum(AtA * (x @ x.T))
        return cp_norm - 2*iprod  # + data norm which is constant
    return loss


def create_gradient(AtA, At_b):
    def grad(x):
        return AtA @ x - At_b
    return grad


def fista_step(x, y, t, lipschitz, smooth_grad_y, l1_reg, prox):
    intermediate_step = (0.5 / lipschitz) * smooth_grad_y

    new_x = prox(y - intermediate_step, l1_reg/lipschitz)
    new_t = 0.5*(1 + np.sqrt(1 + 4 * t**2))
    momentum = (t - 1)/new_t

    dx = new_x - x
    new_y = x + momentum * dx
    return new_x, new_y, new_t


def minimise_fista(lhs, rhs, init, l1_reg, prox, n_iter=10, tol=1e-6, return_err=False):
    """Use the FISTA algorithm to solve the given optimisation problem
    """
    lipschitz = np.trace(lhs) / lhs.shape[0]  # Lower bound for lipschitz

    AtA = lhs
    At_b = rhs
    x = init
    y = init
    t = 1

    compute_smooth_loss = create_loss(AtA, At_b)
    compute_smooth_grad = create_gradient(AtA, At_b)

    loss_x = compute_smooth_loss(x)
    loss_y = loss_x
    smooth_grad_y = compute_smooth_grad(y)
    losses = []
    n_static = 0

    for i in range(n_iter):
        # Simple FISTA update step
        new_x, new_y, new_t = fista_step(x, y, t, lipschitz=lipschitz, smooth_grad_y=smooth_grad_y, l1_reg=l1_reg, prox=prox)
        loss_new_x = compute_smooth_loss(new_x)

        # Adaptive restart criterion from Equation 12 in O’Donoghue & Candès (2012)
        generalised_gradient = y.ravel() - new_x.ravel()
        update_vector = new_x.ravel() - x.ravel()

        # Loss based restart criterion
        if generalised_gradient.T @ update_vector > 0:
            y = x
            smooth_grad_y = compute_smooth_grad(y)
            t = 1
            new_x, new_y, new_t = fista_step(x, y, t, lipschitz=lipschitz, smooth_grad_y=smooth_grad_y, l1_reg=l1_reg, prox=prox)

        # Backtracking line search
        while _should_continue_backtracking(new_x, y, loss_new_x, loss_y, smooth_grad_y, lipschitz):
            lipschitz *= 2
            new_x, new_y, new_t = fista_step(x, y, t, lipschitz=lipschitz, smooth_grad_y=smooth_grad_y, l1_reg=l1_reg, prox=prox)
            loss_new_x = compute_smooth_loss(new_x)

        # Update loop variables
        prev_x = x
        x, y, t = new_x, new_y, new_t
        loss_x = loss_new_x
        loss_y = compute_smooth_loss(y)
        smooth_grad_y = compute_smooth_grad(y)
        losses.append(loss_x)


        if np.linalg.norm(prev_x - x) / np.linalg.norm(x) + 1e-16 < tol:
            n_static += 1
        else:
            n_static = 0
        if n_static > 5:
            break

    if return_err:
        return x, losses
    return x


def fista_solve(lhs, rhs, l1_reg, nonnegative, normalize, init, n_iter_max=100, return_err=False):
    if normalize and l1_reg:
        raise ValueError("Cannot normalize and apply l1 regularization on same mode.")
    
    if l1_reg and nonnegative:
        prox = nn_l1_prox
    elif l1_reg:
        prox = l1_prox
    elif nonnegative and normalize:
        prox = nn_l2ball_prox
    elif nonnegative:
        prox = nn_prox
    elif normalize:
        prox = l2ball_prox

    return minimise_fista(lhs, rhs, init, l1_reg, prox, n_iter=n_iter_max, tol=1e-6, return_err=return_err)
