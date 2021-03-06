from sympy import *
import numpy as np

lm = symbols('lm')
# a = symbols('a')
# lmt = symbols('lmt')
# lm0sina0 = symbols('lm0sina0')
lmt = 0.35  # musculo-tendon len
lm_opt = 0.25  # optimal fiber len
alpha0 = np.pi/4  # pennation angle at optimal fiber len
lt_sl = 0.15
a = 0.5
lm0sina0 = lm_opt*np.sin(alpha0)
kpe = 4.0
kt = 35.0
e0 = 0.6
c1 = 0.2
c2 = 0.995
c3 = 0.25
b11 = 0.815
b12 = 0.433
b13 = 0.1
b21 = 1.055
b22 = 0.717
b23 = 1.0
b31 = 0.162
b32 = -0.03
b33 = 0.354
b41 = 0.063
b42 = 0.2
b43 = 0.0
d1 = -0.318
d2 = -8.149
d3 = -0.374
d4 = 0.886

lt = lmt - sqrt(lm**2 - lm0sina0**2)
cosa = (lmt-lt)/lm
fse = c1*exp(kt*(lt/lt_sl-c2))-c3
fpl = (exp(kpe/e0*(lm/lm_opt-1))-1)/(exp(kpe)-1)
fal = b11*exp((-0.5*(lm/lm_opt-b21)**2)/(b31-b41*lm/lm_opt)**2) + b12*exp((-0.5*(lm/lm_opt-b22)**2)/(b32-b42*lm/lm_opt)**2) + b13*exp((-0.5*(lm/lm_opt-b23)**2)/(b33-b43*lm/lm_opt)**2)
X = (fse/cosa - fpl)/(a*fal)
eq = X - d1*sin(d3) - d4
# reps = [(n, Dummy()) for n in eq.atoms(Float)]
# solve(eq.subs(reps), lm)
print(nsolve(eq, lm, lm_opt))