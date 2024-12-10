# Constantes do problema
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

G = 6.67  # Constante gravitacional fictícia

def three_body(t, y, m1, m2, m3):
    r1 = y[:2]
    r2 = y[2:4]
    r3 = y[4:6]
    v1 = y[6:8]
    v2 = y[8:10]
    v3 = y[10:12]

    r12 = np.linalg.norm(r1 - r2)
    r13 = np.linalg.norm(r1 - r3)
    r23 = np.linalg.norm(r2 - r3)

    a1 = -G * m2 * (r1 - r2) / r12**3 - G * m3 * (r1 - r3) / r13**3
    a2 = -G * m1 * (r2 - r1) / r12**3 - G * m3 * (r2 - r3) / r23**3
    a3 = -G * m1 * (r3 - r1) / r13**3 - G * m2 * (r3 - r2) / r23**3

    return np.concatenate((v1, v2, v3, a1, a2, a3))

# Parâmetros dos corpos
m1, m2, m3 = 130, 130, 130
y0 = [
    -1.3, -1.0, 1.3, -1.0, 0.0, 1.59,
    1.3, 1.3, -1.5, -1.7, 1.6, -1.3
]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 17500)

# Resolver o sistema
sol = solve_ivp(three_body, t_span, y0, args=(m1, m2, m3), t_eval=t_eval, rtol=1e-9, atol=1e-9)

r1, v1 = sol.y[:2], sol.y[6:8]  # Posição e velocidade do Corpo 1

# Detectar cruzamentos onde v_x = 0
crossings_vx = []
tolerance = 0.02  # Tolerância

for i in range(1, len(t_eval)):
    if (v1[0, i-1] * v1[0, i]) < 0:  # Detectar cruzamento
        # Interpolação linear para encontrar cruzamento
        x_cross = r1[0, i-1] + (r1[0, i] - r1[0, i-1]) * (-v1[0, i-1]) / (v1[0, i] - v1[0, i-1])
        y_cross = r1[1, i-1] + (r1[1, i] - r1[1, i-1]) * (-v1[0, i-1]) / (v1[0, i] - v1[0, i-1])
        crossings_vx.append((x_cross, y_cross))

# Separar coordenadas dos cruzamentos
x_cross_vx, y_cross_vx = zip(*crossings_vx)

# Plotar seção de Poincaré
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x_cross_vx, y_cross_vx, c=range(len(x_cross_vx)), cmap='viridis', s=10, alpha=0.9)
plt.colorbar(scatter, label='Ordem dos cruzamentos')
plt.title("Seção de Poincaré (v_x = 0, Corpo 1)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

plt.savefig("poincare_section_vx.png", dpi=300, bbox_inches='tight')
plt.show()
