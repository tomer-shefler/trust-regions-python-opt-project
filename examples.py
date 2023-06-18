import numpy as np

# TEST_FUNC = [f1, f2, f3]

def gradeint_matrix(Q, x):
    return np.array([
        2 * Q[0][0] * x[0] + x[1] * (Q[0][1]+Q[1][0]),
        x[0] * (Q[0][1]+Q[1][0]) + 2 * Q[1][1] * x[1]
    ])

def hessian_matrix(Q, x):
    return np.array([
        [2*x[0], Q[0][1] + Q[1][0]],
        [Q[0][1] + Q[1][0], 2*x[1]]
    ])

def f1(x, should_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f = x.transpose() @ Q @ x
    g = gradeint_matrix(Q, x)
    h = hessian_matrix(Q, x) if should_hessian else 0
    return f, g, h

def f2(x, should_hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f = x.transpose() @ Q @ x
    g = gradeint_matrix(Q, x)
    h = hessian_matrix(Q, x) if should_hessian else 0
    return f, g, h

def f3(x, should_hessian=False):
    Q1 = np.array([[100, 0], [0, 1]])
    Q2 = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q = Q2.transpose() @ Q1 @ Q2
    f = x.transpose() @ Q @ x
    g = gradeint_matrix(Q, x)
    h = hessian_matrix(Q, x) if should_hessian else 0
    return f, g, h

# f(x,y )= 100(y-x^2 )^2+(1-x)^2=100(y^2-2yx^2+x^4 )+1-2x+x^2
# ∇f(x,y) = [-2+2x-400yx+400x^3,200y-200x^2 ]
# ∇^2 f(x,y) [■(2+400y+1200x^2&-400@-400x&200)]
def rosenbrock(x, should_hessian=False):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array([-2 + 2*x[0] - 400*x[0]*x[1] + 400*(x[0]**3), 200*x[1] - 200*(x[0]**2)])
    if should_hessian:
        h = np.array([
            [2 - 400*x[1] + 1200*x[0]**2, -400*x[0]],
            [-400*x[0], 200]
        ])
    else: h = 0
    return f, g, h

def vect(x, should_hessian=False):
    a = np.array([3, 4])
    f = a.T @ x
    g = a
    h = 0
    if should_hessian:
        h = np.zeros((2,2))
    return f, g, h

def e_func(x, should_hessian=False):
    x, y = x[0], x[1]
    e = np.e
    f = e ** (x + 3*y - 0.1) + e ** (x - 3*y - 0.1) + e ** (-x - 0.1)
    g = np.array([
        e ** (x + 3*y - 0.1) + e ** (x - 3*y - 0.1) - e ** (-x - 0.1),
        3*e ** (x + 3*y - 0.1) - 3*e ** (x - 3*y - 0.1)
    ])
    h = 0
    if should_hessian:
        h = np.array([[
            e ** (x + 3*y - 0.1) + e ** (x - 3*y - 0.1) - e ** (-x - 0.1),
            3*e ** (x + 3*y - 0.1) - 3*e ** (x - 3*y - 0.1)
        ], [
            3*e ** (x + 3*y - 0.1) - 3*e ** (x - 3*y - 0.1),
            9*e ** (x + 3*y - 0.1) + 9*e ** (x - 3*y - 0.1),
        ]])

    return f, g, h
