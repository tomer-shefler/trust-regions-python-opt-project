import numpy as np
import examples


def dogleg(H, g, B, trust_radius):
    pb = -H@g                    # full newton step
    norm_pb = np.linalg.norm(pb)
    
    # full newton step lies inside the trust region
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
    tau = (-pu_pb_pu_sq + np.sqrt(d)) / pb_pu_sq+1    # 0<tau<1
    if tau < 1:
        return pu*tau
    # 1<tau<2
    return pu + (tau-1) * pb_pu

def trust_region(x0, eta=0.15, tol=1e-4, max_trust_radius=1.0):
    xx = [] # to store the iterates
    #initial point
    x = x0
    r = [] # to store the trust radius at each iteration
    #initial radius
    trust_radius = 0.1
    r.append(trust_radius)
    xx.append(x)
    while True:
        f, g, B = examples.rosenbrock(x, should_hessian=True)
        H = np.linalg.inv(B)
        p = dogleg(H, g, B, trust_radius) ##me
        f_x_p, _, _ = examples.rosenbrock(x + p)
        rho = (f - f_x_p)/(-(np.dot(g, p) + 0.5 * np.dot(p, B@p)))
        norm_p = np.linalg.norm(p)
        if rho < 0.25:
            trust_radius = 0.25 * norm_p
        else:
            if rho > 0.75 and norm_p == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius
        # r.append(trust_radius)
        if rho > eta:
            x = x + p
        xx.append((x, f))
        if np.linalg.norm(g) < tol:
            break
    return xx, r


if __name__ == "__main__":
    l, r = trust_region(np.array([5,5]))
    print(l)