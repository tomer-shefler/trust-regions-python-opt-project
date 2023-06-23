import numpy as np
import examples

class TrustRegion(object):
    def __init__(self, eta=0.15, tol=1e-4, max_trust_radius=1.0):
        self.eta = eta
        self.tol = tol
        self.max_trust_radius = 1.0

    def dogleg(self, g, B, trust_radius):
        # full newton step lies inside the trust region
        pb = -np.linalg.inv(B)@g
        if np.linalg.norm(pb) <= trust_radius:
            return pb

        # step along the steepest descent direction lies outside the  
        # trust region       
        pu = - (np.dot(g, g) / np.dot(g, B@g)) * g
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
            return pu*tau
        return pu + (tau-1) * pb_pu

    def trust_region(self, x0):
        record = []
        #initial point
        x = x0
        #initial radius
        trust_radius = 0.1
        f, _, _ = examples.rosenbrock(x)
        record.append((x, f))
        while True:
            f, g, B = examples.rosenbrock(x, should_hessian=True)
            H = np.linalg.inv(B)
            p = self.dogleg(g, B, trust_radius) ##me
            f_x_p, _, _ = examples.rosenbrock(x + p)
            rho = (f - f_x_p)/(-(np.dot(g, p) + 0.5 * np.dot(p, B@p)))
            norm_p = np.linalg.norm(p)
            if rho < 0.25:
                trust_radius = 0.25 * norm_p
            else:
                if rho > 0.75 and norm_p == trust_radius:
                    trust_radius = min(2.0 * trust_radius, self.max_trust_radius)
                else:
                    trust_radius = trust_radius
            if rho > self.eta:
                x = x + p
            record.append((x, f))
            if np.linalg.norm(g) < self.tol:
                break
        return record


if __name__ == "__main__":
    t = TrustRegion()
    x0 = np.array([5,5])
    l = t.trust_region(x0)
    print(l)