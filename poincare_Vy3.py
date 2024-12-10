import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constante gravitacional fictícia (ajustada para evitar números extremamente pequenos)
G = 6.67

# Função que define o sistema de equações diferenciais
def three_body(t, y, m1, m2, m3):
    r1 = y[:2]
    r2 = y[2:4]
    r3 = y[4:6]
    v1 = y[6:8]
    v2 = y[8:10]
    v3 = y[10:12]

    # Distâncias entre os corpos
    r12 = np.linalg.norm(r1 - r2)
    r13 = np.linalg.norm(r1 - r3)
    r23 = np.linalg.norm(r2 - r3)

    # Acelerações devido à gravidade
    a1 = -G * m2 * (r1 - r2) / r12**3 - G * m3 * (r1 - r3) / r13**3
    a2 = -G * m1 * (r2 - r1) / r12**3 - G * m3 * (r2 - r3) / r23**3
    a3 = -G * m1 * (r3 - r1) / r13**3 - G * m2 * (r3 - r2) / r23**3

    return np.concatenate((v1, v2, v3, a1, a2, a3))

# Parâmetros dos corpos (massas fictícias)
m1, m2, m3 = 130, 130, 130

# Condições iniciais
y0 = [
    -1.3, -1.0,  # Posição do corpo 1
     1.3, -1.0,  # Posição do corpo 2
     0.0, 1.59,  # Posição do corpo 3
     1.3, 1.3,   # Velocidade do corpo 1
    -1.5, -1.7,  # Velocidade do corpo 2
     1.6, -1.3   # Velocidade do corpo 3
]

# Intervalo de tempo e pontos de avaliação
t_span = (0, 50)  # Por exemplo, aumente de 20 para 100 ou mais.
t_eval = np.linspace(*t_span, 3000)  # Mais pontos para análise precisa.

# Resolver o sistema
sol = solve_ivp(three_body, t_span, y0, args=(m1, m2, m3), t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Extrair dados para o corpo 2
r2, v2 = sol.y[2:4], sol.y[8:10]

# Identificar cruzamentos onde v_y do corpo 2 = 0
crossings = []

for i in range(1, len(t_eval)):
    # Verificar se v_y cruza 0
    if v2[1, i-1] * v2[1, i] < 0:
        # Interpolação linear para encontrar o cruzamento
        x_cross = r2[0, i-1] + (r2[0, i] - r2[0, i-1]) * (-v2[1, i-1]) / (v2[1, i] - v2[1, i-1])
        y_cross = r2[1, i-1] + (r2[1, i] - r2[1, i-1]) * (-v2[1, i-1]) / (v2[1, i] - v2[1, i-1])
        crossings.append((x_cross, y_cross))

# Separar as coordenadas dos cruzamentos
x_cross, y_cross = zip(*crossings)


# Plotar a seção de Poincaré
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x_cross, y_cross, c=range(len(x_cross)), cmap='plasma', s=10, alpha=0.9)
plt.colorbar(scatter, label='Ordem dos cruzamentos')
plt.title("Seção de Poincaré (v_y = 0, Corpo 3)")

plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

plt.savefig("poincare_section_vy0_body3.png", dpi=300, bbox_inches='tight')
plt.show()
