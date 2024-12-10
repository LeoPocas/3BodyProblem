import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constante gravitacional fictícia (ajustada para evitar números extremamente pequenos)
G = 6.67
y_target = -10.0

def on_click(event):
    print(f"Você clicou em: x = {event.xdata}, y = {event.ydata}")

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
     1.3, 1.3,  # Velocidade do corpo 1
    -1.5, -1.7, # Velocidade do corpo 2
     1.6, -1.3    # Velocidade do corpo 3
]


# Intervalo de tempo e pontos de avaliação
t_span = (0, 50)  # Por exemplo, aumente de 20 para 100 ou mais.
t_eval = np.linspace(*t_span, 3000)  # Mais pontos para análise precisa.

# Resolver o sistema
sol = solve_ivp(three_body, t_span, y0, args=(m1, m2, m3), t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Extrair dados
r1, v1 = sol.y[:2], sol.y[6:8]

# Identificar cruzamentos com o plano y=0
crossings = []
tolerance = 0.05  # Tolerância em torno de y_target

for i in range(1, len(t_eval)):
    # Verificar se y cruza y_target com tolerância
    if (r1[1, i-1] - y_target) * (r1[1, i] - y_target) < 0:
        # Interpolação linear para encontrar o cruzamento
        x_cross = r1[0, i-1] + (r1[0, i] - r1[0, i-1]) * (y_target - r1[1, i-1]) / (r1[1, i] - r1[1, i-1])
        vx_cross = v1[0, i-1] + (v1[0, i] - v1[0, i-1]) * (y_target - r1[1, i-1]) / (r1[1, i] - r1[1, i-1])
        crossings.append((x_cross, vx_cross))

# Separar as coordenadas dos cruzamentos
x_cross, vx_cross = zip(*crossings)

# Plotar a seção de Poincaré
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x_cross, vx_cross, c=range(len(x_cross)), cmap='viridis', s=10, alpha=0.9)
plt.colorbar(scatter, label='Ordem dos cruzamentos')
plt.title("Seção de Poincaré (y = 0, Corpo 1)")

plt.xlabel("x")
plt.ylabel("vx")
plt.xlim(5, 11) 
plt.ylim(-25, 25)
plt.grid(True)

plt.savefig("poincare_section_y_target-10.png", dpi=300, bbox_inches='tight')
plt.show()
