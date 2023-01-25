import numpy as np
import scipy.linalg as sla


def l1_prox(x, reg):
    """Proximal operator to apply l1 sparsity penalty. 
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array. 
    reg : float
        L1 sparsity penalty (reg >= 0). 
        
    Returns
    -------
    x : numpy.ndarray
        Input array with l1 proximal operation applied. 
    """
    sign = np.sign(x)
    return sign * np.maximum(0, np.abs(x) - reg)


def nn_prox(x, reg):
    """Proximal operator to apply non-negativity constraint. 
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array. 
    reg : float
        Not necessary, but included for continuity of the proximal operator
        call signature. 
        
    Returns
    -------
    x : numpy.ndarray
        Input array with non-negativity constraint applied.
    """
    return np.maximum(0, x)


def l2ball_prox(x, reg):
    """Proximal operator to apply l2 normalization constraint. Constrains
    l2-norm of `x` to be less than or equal to 1.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array. 
    reg : float
        Not necessary, but included for continuity of the proximal operator
        call signature. 
        
    Returns
    -------
    x : numpy.ndarray
        Input array with l2 norm constraint applied.
    """
    return x / np.maximum(1, np.linalg.norm(x, axis=1, keepdims=True))


def nn_l1_prox(x, reg):
    """Proximal operator to apply nonnegative constraint in combination 
    with an l1 sparsity penalty. 
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array. 
    reg : float
        L1 sparsity penalty (reg >= 0). 
        
    Returns
    -------
    x : numpy.ndarray
        Input array with non-negativity constraint and l1 proximal operation 
        applied.
    """
    return np.maximum(0, x - reg)


def nn_l2ball_prox(x, reg):
    """Proximal operator to apply nonnegative constraint in combination 
    with an l2 normalization constraint. 
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array. 
    reg : float
        Not necessary, but included for continuity of the proximal operator
        call signature.
        
    Returns
    -------
    x : numpy.ndarray
        Input array with non-negativity and l2 norm constraints applied.
    """
    return l2ball_prox(nn_prox(x, None), None)


def _should_continue_backtracking(
    new_x, 
    y, 
    loss_new_x, 
    loss_y, 
    smooth_grad_y, 
    lipschitz
):
    """Determines whether or not a backtracking line search to decrease the
    step size should continue. Based on the 'FISTA with backtracking' 
    algorithm outlined in Beck & Teboulle (2009). 
    
    If the step length is large, then a large decrease in loss should also be 
    expected. If instead a large step results in a relatively small decrease in 
    the loss, then the step is likely overshooting the target. To prevent this, 
    the algorithm starts with a large guess for the step length, and decreases 
    it until there is a sufficient decrease in the loss. This allows for
    maximal initial step sizes to be refined as necessary, resulting in a 
    speed up in the optimization.
    
    Parameters
    ----------
    new_x : numpy.ndarray
        Updated solution vector `x`. 
    y : numpy.ndarray
        Momentum vector `y`. 
    loss_new_x : numpy.ndarray
        Loss resulting from updated solution vector `x`. 
    loss_y : numpy.ndarray
        Loss of momentum vector `y`. 
    smooth_grad_y : numpy.ndarray
        Gradient of momentum vector `y`. 
    lipschitz : float
        Lipschitz coefficient.
        
    Returns
    -------
    continue_backtracking : bool
        If True, indicates backtracking line search should continue.
    """
    update_vector = new_x - y
    update_distance = np.sum(update_vector**2) * lipschitz / 2.5
    linearised_improvement = smooth_grad_y.ravel().T @ update_vector.ravel()
    continue_backtracking = loss_new_x - loss_y > update_distance + linearised_improvement
    return continue_backtracking


def create_loss(AtA, At_b):
    """Helper function to generate loss function.
    
    Parameters
    ----------
    AtA : numpy.ndarray
        The matrix A^T A.
    At_b : numpy.ndarray
        The vector A^T b.
        
    Returns
    -------
    loss : function
        Loss function.
    """
    def loss(x):
        iprod = np.sum(At_b * x)
        cp_norm = np.sum(AtA * (x @ x.T))
        return 0.5 * (cp_norm - 2 * iprod)  # + data norm which is constant
    return loss


def create_gradient(AtA, At_b):
    """Helper function to generate gradient function.
    
    Parameters
    ----------
    AtA : numpy.ndarray
        The matrix A^T A.
    At_b : numpy.ndarray
        The vector A^T b.
        
    Returns
    -------
    grad : function
        Gradient function.
    """
    def grad(x):
        return AtA @ x - At_b
    return grad


def fista_step(
    x, 
    y, 
    t, 
    lipschitz, 
    smooth_grad_y, 
    l1_reg, 
    prox
):
    """Function to take one FISTA step.
    
    Parameters
    ----------
    x : numpy.ndarray
        Initial solution vector `x`. 
    y : numpy.ndarray 
        Initial solution vector `x` with momentum.
    t : float
        Momentum coefficient.
    lipschitz : float
        Lipschitz coefficient.
    smooth_grad_y : numpy.ndarray
        Gradient of `y`. 
    l1_reg : float
        L1 regularization coefficient. 
    prox : function
        Proximal operator.
        
    Returns
    -------
    new_x : numpy.ndarray
        Updated solution vector `x`.
    new_y : numpy.ndarray
        Updated solution vector `x` with momentum.
    new_t : float
        Updated momentum coefficient.
    """
    intermediate_step = (1 / lipschitz) * smooth_grad_y
    new_x = prox(y - intermediate_step, l1_reg / lipschitz)
    new_t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
    momentum = (t - 1) / new_t
    dx = new_x - x
    new_y = x + momentum * dx
    return new_x, new_y, new_t


def minimise_fista(
    lhs,
    rhs,
    init,
    l1_reg,
    prox,
    n_iter=10,
    tol=1e-6,
    return_err=False,
    line_search=True,
):
    """Use the FISTA algorithm to solve the given optimisation problem:
    
        min_x ||Ax - b||^2 + reg * g(x)
        
    Optimization is acheived using the Fast Iterative Shrinkage Thresholding
    Algorithm (FISTA) with backtracking, as described in Beck & Teboulle (2009), 
    in combination with adaptive restart as described in O'Donoghue & Candès (2012).
    
    Parameters
    ----------
    lhs : numpy.ndarray
        The matrix A^T A. 
    rhs : numpy.ndarray
        The vector A^T b.
    init : numpy.ndarray
        Initialization of `x`.
    l1_reg : float
        L1 regularization coefficient `reg`.
    prox : function
        The proximal operator `g()`.
    n_iter : int, default is 10
        Maximal number of iterations.
    tol : float, default is 1e-6
        Convergence tolerance.
    return_err : bool, default is False
        Return iteration errors if true.
    line_search : bool, default is True
        Perform backtracking line search if True. 
    
    Returns
    -------
    x : numpy.ndarray
        The solution `x` that minimizes the given optimization problem.
    losses : list
        If `return_err`=True, a list of iteration errors.
    """
    losses = [None] * n_iter
    # if provided data is all zeros, don't run fista, just return zero matrix
    if np.linalg.norm(lhs) == 0 or np.linalg.norm(rhs) == 0:
        x = np.zeros_like(init)
        if return_err:
            return x, losses[:0]
        return x

    A_norm = np.trace(lhs)
    if line_search:
        lipschitz = np.trace(lhs) / (2 * lhs.shape[0])  # Lower bound for lipschitz
    else:
        lipschitz = np.trace(lhs)  # Lower bound for lipschitz

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
    n_static = 0

    for i in range(n_iter):
        # Simple FISTA update step
        new_x, new_y, new_t = fista_step(
            x,
            y,
            t,
            lipschitz=lipschitz,
            smooth_grad_y=smooth_grad_y,
            l1_reg=l1_reg,
            prox=prox,
        )
        loss_new_x = compute_smooth_loss(new_x)

        # Adaptive restart criterion from Equation 12 in O’Donoghue & Candès (2012)
        # If the gradient+prox update without momentum points in the opposite 
        # direction of the full update, then the momentum is likely to push the 
        # estimate in the wrong direction, in which case we restart the momentum.
        if loss_new_x > loss_x:
            y = x
            smooth_grad_y = compute_smooth_grad(y)
            t = 1
            new_x, new_y, new_t = fista_step(
                x,
                y,
                t,
                lipschitz=lipschitz,
                smooth_grad_y=smooth_grad_y,
                l1_reg=l1_reg,
                prox=prox,
            )
            loss_new_x = compute_smooth_loss(new_x)

        # Backtracking line search
        for line_search_it in range(5):
            if (
                not _should_continue_backtracking(
                    new_x, y, loss_new_x, loss_y, smooth_grad_y, lipschitz
                )
                or not line_search
            ):
                break
            lipschitz *= 1.5
            new_x, new_y, new_t = fista_step(
                x,
                y,
                t,
                lipschitz=lipschitz,
                smooth_grad_y=smooth_grad_y,
                l1_reg=l1_reg,
                prox=prox,
            )
            loss_new_x = compute_smooth_loss(new_x)

        # Update loop variables
        prev_x = x
        x, y, t = new_x, new_y, new_t
        loss_x = loss_new_x
        loss_y = compute_smooth_loss(y)
        smooth_grad_y = compute_smooth_grad(y)
        losses[i] = loss_x

        if np.linalg.norm(prev_x - x) * A_norm < tol:
            n_static += 1
        else:
            n_static = 0

        # break after 5 static iterations
        if n_static > 5:
            break

    if return_err:
        return x, losses[: i + 1]
    return x


def fista_solve(
    lhs, 
    rhs, 
    l1_reg, 
    nonnegative, 
    normalize, 
    init, 
    n_iter_max=100, 
    return_err=False
):
    """Use the FISTA algorithm to define and solve the optimisation problem:
    
        min_x ||Ax - b||^2 + reg * g(x)
    
    Where `reg` is an l1 regularisation coefficient, and `g(x)` is a proximal 
    operator applied to `x`. The proximal operator can optionally incorporate 
    (individually or in combination): 
        a) l1 regularization
        b) non-negativity constraint
        c) l2 norm constraint: ||x|| <= 1
    L1 regularization and l2 norm constraint cannot be applied in combination.
    
    Optimization is acheived using the Fast Iterative Shrinkage Thresholding
    Algorithm (FISTA) with backtracking, as described in Beck & Teboulle (2009), 
    in combination with adaptive restart as described in O'Donoghue & Candès (2012).
    
    Parameters
    ----------
    lhs : numpy.ndarray
        The matrix A^T A. 
    rhs : numpy.ndarray
        The vector A^T b.
    l1_reg : float
        L1 regularization coefficient `reg`.
    nonnegative : bool
        If True, applies a non-negativity constraint to the solution `x`.
    normalize : bool
        If True, applies an l2 norm constraint to `x` such that ||x|| <= 1.
    init : numpy.ndarray
        Initialization of `x`.
    n_iter_max : int, default is 100
        Maximal number of iterations.
    return_err : bool, default is False
        Return iteration errors if true.
    
    Returns
    -------
    x : numpy.ndarray
        The solution `x` that minimizes the given optimization problem.
    losses : list
        If `return_err`=True, a list of iteration errors.
    """
    if normalize and l1_reg:
        raise ValueError('Cannot normalize and apply l1 regularization on same mode.')

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

    return minimise_fista(
        lhs, rhs, init, l1_reg, prox, n_iter=n_iter_max, tol=1e-6, return_err=return_err
    )
