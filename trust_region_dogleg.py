import numpy as np
import examples
import function_graph_draw


class TrustRegion(object):
    def __init__(self, eta=0.15, tol=1e-4, max_trust_radius=1, max_iter=100000):
        self.eta = eta
        self.tol = tol
        self.max_trust_radius = max_trust_radius
        self.max_iter = max_iter

    def dogleg(self, g, B, trust_radius):
        # full newton step lies inside the trust region
        pb = 0
        if np.count_nonzero(B) > 0:
            pb = -np.linalg.inv(B) @ g
        if trust_radius >= np.linalg.norm(pb) > 0:
            return pb

        # step along the steepest descent direction lies outside the
        # trust region
        pu = -(np.dot(g, g) / np.dot(g, B @ g)) * g
        dot_pu = np.dot(pu, pu)
        norm_pu = np.sqrt(dot_pu)
        if norm_pu >= trust_radius:
            return trust_radius * pu / norm_pu
        # solve ||pu**2 +(tau-1)*(pb-pu)**2|| = trust_radius**2
        pb_pu = pb - pu
        pb_pu_sq = np.dot(pb_pu, pb_pu)
        pu_pb_pu_sq = np.dot(pu, pb_pu)
        d = pu_pb_pu_sq ** 2 - pb_pu_sq * (dot_pu - trust_radius ** 2)
        tau = (-pu_pb_pu_sq + np.sqrt(d)) / pb_pu_sq + 1
        if tau < 1:
            return pu * tau
        return pu + (tau - 1) * pb_pu

    def trust_region(self, x0, f):
        record = []
        # initial point
        x = x0
        # initial radius
        trust_radius = 0.1
        f_x, _, _ = f(x)
        record.append((x, f_x))
        gk = 0
        g_prev = 0
        for _ in range(self.max_iter):
            f_x, g, B = f(x, should_hessian=True)
            g_prev = np.linalg.norm(gk)
            gk = np.linalg.norm(g)
            p = self.dogleg(g, B, trust_radius)
            f_x_p, _, _ = f(x + p)
            rho = (f_x - f_x_p) / (-(np.dot(g, p) + 0.5 * np.dot(p, B @ p)))
            norm_p = np.linalg.norm(p)
            if rho < 0.25:
                trust_radius = 0.25 * trust_radius
                if rho < self.eta:
                    continue
            else:
                if rho > 0.75 and norm_p == trust_radius:
                    trust_radius = min(2.0 * trust_radius, self.max_trust_radius)
            x = x + p
            record.append((x, f_x))
            if np.linalg.norm(g) < self.tol:
                break
        return record


if __name__ == "__main__":
    t = TrustRegion()
    x0 = np.array([-1, 2])
    l = t.trust_region(x0, examples.rosenbrock)
    graph_drawer = function_graph_draw.GraphDrawer(examples.rosenbrock, graph_range=2)
    for slot in l:
        x, f_x = slot
        point = [x[0], x[1], f_x]
        graph_drawer.draw_point(point)
    graph_drawer.finish_draw()
