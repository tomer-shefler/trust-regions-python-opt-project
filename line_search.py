import numpy as np

class UnconstrainedMin(object):
    C1 = 0.01
    BACKTRACKING = 0.5

    def __init__(self, obj_tol=1e-8 ,param_tol=1e-12):
        self._obj_tol = obj_tol
        self._param_tol = param_tol
        self.minimizers = [
            self.gradient_descent,
            self.newton,
            self.bgfs,
            self.sr1
        ]

    def wolfe_cond(self, f_x, g_x, f_x_next, p, alpha):
        return f_x_next <= f_x + self.C1 * alpha * g_x.T @ p

    def find_next(self, x, f, p):
        """
        Find alpha with wolfe conds and return x_next and f(x_next)
        """
        alpha = 1
        wolfe_conds_set = False
        f_x, g_x, _ = f(x)
        while not wolfe_conds_set:
            x_next = x + alpha * p
            f_x_next, g_x_next, _ = f(x_next)
            wolfe_conds_set = self.wolfe_cond(f_x, g_x, f_x_next, p, alpha)
            alpha *= self.BACKTRACKING
            if alpha == 0:
                raise Exception("alpha is zero")

        return x_next, f_x_next, g_x_next

    def check_tol(self, x, x_next, f_x, f_x_next):
        if np.linalg.norm(x - x_next) < self._param_tol:
            return True
        if np.abs(f_x - f_x_next) < self._obj_tol:
            return True
        return False

    def line_search_min(self, minimizer, f, x0, max_iter=100):
        iter = 0
        success = False
        x = x0
        b = np.eye(len(x))
        record = []
        while not success and iter < max_iter:
            f_x, _, _ = f(x)
            x_next, f_x_next, b = minimizer(f, x, b)
            success = self.check_tol(x, x_next, f_x, f_x_next)
            print(f"Iteration {iter}: x={x}, f(x)={f_x}")
            record.append((x, f_x))
            x = x_next
            iter += 1
        
        return success, x, f_x, record

    def gradient_descent(self, f, x, *args):
        f_x, g_x, _ = f(x)
        p = -1 * g_x
        x_next, f_x_next, g_x_next = self.find_next(x, f, p)
        return x_next, f_x_next, 0

    def newton(self, f, x, *args):
        f_x, g_x, h_x = f(x, should_hessian=True)
        p = -1 * np.linalg.inv(h_x) @ g_x
        x_next, f_x_next, g_x_next = self.find_next(x, f, p)
        return x_next, f_x_next, 0
        
    def bgfs(self, f, x, b):
        f_x, g_x, _ = f(x)
        p = -b @ g_x
        x_next, f_x_next, g_x_next = self.find_next(x, f, p)
        s = x_next - x
        y = g_x_next - g_x
        b += -1 * (b @ s @ s.T * b) / (s.T @ b @ s)
        b += (y @ y.T) / (y.T @ s)
        return x_next, f_x_next, b

    def sr1(self, f, x, b):
        f_x, g_x, _ = f(x)
        p = -b @ g_x
        x_next, f_x_next, g_x_next = self.find_next(x, f, p)
        s = x_next - x
        y = g_x_next - g_x
        y_minus_bs = y - b @ s
        b += (y_minus_bs @ y_minus_bs.T) / (y_minus_bs.T @ s)
        return x_next, f_x_next, b
