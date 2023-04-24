from sympy.abc import alpha, beta, gamma
from sympy import symbols, Array, Matrix
from sympy import derive_by_array, simplify

u, v = symbols('u v')
h1, h2, h3 = symbols('h1 h2 h3')
g1, g2, g3 = symbols('g1 g2 g3')

h1_vec = Matrix([[h1], [h2], [h3]])
h2_vec = Matrix([[g1], [g2], [g3]])

K_mat = Matrix([[alpha, gamma, u], [0, beta, v], [0, 0, 1]])
K_mat_inv = K_mat.inv()

B_mat = simplify(K_mat_inv.transpose()*K_mat_inv * (alpha*beta)**2)

f1 = h1_vec.transpose()*B_mat*h2_vec
kl = simplify(f1[0,0])
f2 = h1_vec.transpose()*B_mat*h1_vec - h2_vec.transpose()*B_mat*h2_vec

df1 = simplify(derive_by_array(simplify(f1[0,0]), [alpha, beta, gamma, u, v]))
df2 = simplify(derive_by_array(simplify(f2[0,0]), [alpha, beta, gamma, u, v]))

print(f1)
print(f2)
print('finished')