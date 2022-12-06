import sympy as smp
import numpy as np

if __name__ == '__main__':
    tQ = 1
    Jc = 1
    dt = 0.0001
    Jc_s, tQ_s, t_s, t0_s, dt_s, x = smp.symbols('J_c t_Q t t_0 dt x', real=True)
    ramp = Jc_s * (1 + t_s / tQ_s - smp.Rational(4, 27) * t_s ** 3 / tQ_s ** 3)
    linear = Jc_s * (1 + t_s / tQ_s)
    # H = smp.Heaviside
    H = smp.Piecewise((0,t_s<0),(1,t_s>0),(1/2,True))
    aa = 3 * (1 - H) + 6 * H
    print(smp.simplify(aa))
    aal = smp.lambdify(t_s, aa)
    print("J_average(-1) =", aal(np.array(-1)))
    print("J_average(0) =", aal(np.array(0)))
    print("J_average(1) =", aal(np.array(1)))