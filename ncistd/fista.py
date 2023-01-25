import numpy as np
import scipy.linalg as sla


def l1_prox(x, reg):
    """Proximal operator to apply l1 sparsity penalty. 
    
    Parameters
    ----------
    x : numpy.array
        Input array. 
    reg : float
        L1 sparsity penalty (reg >= 0). 
        
    Returns
    -------
    x : numpy.array
        Input array with l1 proximal operation applied. 
    """
    sign = np.sign(x)
    return sign * np.maximum(0, np.abs(x) - reg)

def nn_prox(x, reg):
    """Proximal operator to apply non-negativity constraint. 
    
    Parameters
    ----------
    x : numpy.array
        Input array. 
    reg : float
        Not necessary, but included for continuity of the proximal operator
        call signature. 
        
    Returns
    -------
    x : numpy.array
        Input array with non-negativity constraint applied.
    """
    return np.maximum(0, x)


def l2ball_prox(x, reg):
    """Proximal operator to apply l2 normalization constraint. Constrains
    l2-norm of the rows of x to be less than or equal to 1.
    
    Parameters
    ----------
    x : numpy.array
        Input array. 
    reg : float
        Not necessary, but included for continuity of the proximal operator
        call signature. 
        
    Returns
    -------
    x : numpy.array
        Input array with l2 norm constraint applied.
    """
    return x / np.maximum(1, np.linalg.norm(x, axis=1, keepdims=True))


def nn_l1_prox(x, reg):
    """Proximal operator to apply nonnegative constraint in combination 
    with an l1 sparsity penalty. 
    
    Parameters
    ----------
    x : numpy.array
        Input array. 
    reg : float
        L1 sparsity penalty (reg >= 0). 
        
    Returns
    -------
    x : numpy.array
        Input array with non-negativity constraint and l1 proximal operation 
        applied.
    """
    return np.maximum(0, x - reg)


def nn_l2ball_prox(x, reg):
    """Proximal operator to apply nonnegative constraint in combination 
    with an l2 normalization constraint. 
    
    Parameters
    ----------
    x : numpy.array
        Input array. 
    reg : float
        Not necessary, but included for continuity of the proximal operator
        call signature.
        
    Returns
    -------
    x : numpy.array
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
    """Based on FISTA with backtracking.
    
    Parameters
    ----------
    new_x : numpy.array
    y : numpy.array  
    loss_new_x : numpy.array  
    loss_y : numpy.array   
    smooth_grad_y : numpy.array   
    lipschitz : float
        
    Returns
    -------
    continue_backtracking : bool
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
    At_b : numpy.ndarray
        
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
    """Helper function to gradient function.
    
    Parameters
    ----------
    AtA : numpy.ndarray
    At_b : numpy.ndarray
        
    Returns
    -------
    grad : function
        Gradient function.
    """
    def grad(x):
        return AtA @ x - At_b
    return grad


def fista_step(x, y, t, lipschitz, smooth_grad_y, l1_reg, prox):
    """Function to take one FISTA step.
    
    Parameters
    ----------
    x : numpy.array
    y : numpy.array 
    t : float
    lipschitz : float
    smooth_grad_y numpy.array
    l1_reg : float
    prox : function
        
    Returns
    -------
    new_x : numpy.array
    new_y : numpy.array
    new_t : float
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
    """Use the FISTA algorithm to solve the given optimisation problem
    
    Parameters
    ----------
    lhs : numpy.array
    rhs : numpy.array
    init : numpy.array
    l1_reg : float
    prox : function
    n_iter : int, default is 10
    tol : float, default is 1e-6
    return_err : bool, default is False
    line_search : bool, default is True
    
    Returns
    -------
    x : numpy.array
    losses : list
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
        # Loss based restart criterion
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
    """Use the FISTA algorithm to solve the given optimisation problem
    
    Parameters
    ----------
    lhs : numpy.array
    rhs : numpy.array
    l1_reg : float
    nonnegative : bool
    normalize : bool
    init : numpy.array
    n_iter_max : int, default is 100
    return_err : bool, default is False
    
    Returns
    -------
    x : numpy.array
    losses : list
    """
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

    return minimise_fista(
        lhs, rhs, init, l1_reg, prox, n_iter=n_iter_max, tol=1e-6, return_err=return_err
    )
