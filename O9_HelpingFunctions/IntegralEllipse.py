import sympy as sp
from sympy.solvers.inequalities import reduce_inequalities

x, y, z = sp.symbols('x y z')
x0, y0, z0 = sp.symbols('x0 y0 z0')
vx, vy, vz = sp.symbols('vx vy vz')
r = sp.symbols('r')
t = sp.symbols('t')
u, v = sp.symbols('u v')

eqx = x0 + vx*t
eqy = y0 + vy*t
eqz = z0 + vz*t

#eqy = sp.Eq(y - y0 - vy*t, 0)
#eqz = sp.Eq(z - z0 - vz*t, 0)

inD = sp.Ge((u*x + v*y + z)**2 - (u**2 + v**2 + 1)*(x**2 + y**2 + z**2 - r**2),0)
inD = inD.subs(x, eqx)
inD = inD.subs(y, eqy)
inD = inD.subs(z, eqz)
inD = sp.simplify(inD)
#reduce_inequalities(inD, t)

a0, a1, a2 = sp.symbols('a0 a1 a2')
Dt = sp.symbols('Dt')

reduce_inequalities([a2 * t**2 + 2 * a1 * t + a0 >= 0, 0 <= t, Dt >= t], [t])
#sp.solve([eqx, eqy, eqz, eqD], t)
# Descrimante has to be positive

# Task: Search the time t, where the descriminante is positive for image coordinates (u,v)
