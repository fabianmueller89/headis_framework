import sympy as sp
from sympy.solvers.inequalities import reduce_inequalities

### Calculates world coordinates dependend on ellipse middle point, radius and z-deepth coordinate

x0, y0, z0 = sp.symbols('x0 y0 z0') #Ball center point in 3d-world coordinates
x, y, z = sp.symbols('x y z') #Ball center point in 3d-world coordinates
r = sp.symbols('r')  # Radius of Ball

# Middle point of ellipse in frame coordinates
u, v = sp.symbols('u v')

t = sp.symbols('t')  # Radius of Ball

row_one = sp.Eq((r**2 - y0**2 - z0**2)*u + x0*y0*v, -x0*z0)
row_two = sp.Eq((r**2 - x0**2 - z0**2)*v + x0*y0*u, -y0*z0)

#row_one = sp.Eq((r**2 - y0**2*z0**2 - z0**2)*u + x0*y0*v*z0**2, -x0*z0**2)
#row_two = sp.Eq((r**2 - x0**2*z0**2 - z0**2)*v + x0*y0*u*z0**2, -y0*z0**2)

solution = sp.solve([row_one, row_two], (x0, y0)) # in dependency of x0 and y0 to infer on proper coordinates

### Provides three points (x,y)
print(solution[0])
print(solution[1])
print(solution[2])

### To check, which of the solution is correct, the beam through the ellipse (with z=1) has to intersect the ball in 3D
ball = sp.Le((z * u - x0)**2 - (z * v - y0)**2 - (z - z0)**2, r**2)

idx=0
ball = ball.subs(x0, solution[idx][0])
ball = ball.subs(y0, solution[idx][1])
print(ball)
#eq_z = sp.Le(z, 1)
reduce_inequalities([ball], (z))