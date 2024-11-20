import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constante gravitacional fictícia (ajustada para evitar números extremamente pequenos)
G = 1e1
MAX_ACCELERATION = 10

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

    a1 = np.clip(a1, -MAX_ACCELERATION, MAX_ACCELERATION)
    a2 = np.clip(a2, -MAX_ACCELERATION, MAX_ACCELERATION)
    a3 = np.clip(a3, -MAX_ACCELERATION, MAX_ACCELERATION)

    return np.concatenate((v1, v2, v3, a1, a2, a3))

# Parâmetros dos corpos (massas fictícias)
m1, m2, m3 = 2.0, 2.0, 2.0

# Condições iniciais
y0_base = [
    -1.5, 0.0,  # Posição do corpo 1
     1.5, 0.0,  # Posição do corpo 2
     0.0, 2.59,  # Posição do corpo 3
     1.4, 1.4,  # Velocidade do corpo 1
    -1.5, -1.7, # Velocidade do corpo 2
     2, -1.6  # Velocidade do corpo 3
]

# Intervalo de tempo e pontos de avaliação
t_span = (0, 20)
t_eval = np.linspace(*t_span, 1000)

# Armazenar resultados de todas as simulações
all_routes = []

# Simular 10 cenários diferentes
for i in range(10):
    perturbation = np.random.normal(0, 1e-3, len(y0_base))
    y0_perturbed = np.array(y0_base) + perturbation

    sol = solve_ivp(three_body, t_span, y0_perturbed, args=(m1, m2, m3), t_eval=t_eval, rtol=1e-9, atol=1e-9)

    r1, r2, r3 = sol.y[:2], sol.y[2:4], sol.y[4:6]
    
    # Guardar rotas
    all_routes.append({
        'r1': r1,
        'r2': r2,
        'r3': r3
    })

# Criar um gráfico das rotas personalizadas
fig, ax = plt.subplots(figsize=(12, 12))

colors_body1 = plt.cm.Blues(np.linspace(0.4, 1, len(all_routes)))
colors_body2 = plt.cm.Greens(np.linspace(0.4, 1, len(all_routes)))
colors_body3 = plt.cm.Reds(np.linspace(0.4, 1, len(all_routes)))

for idx, routes in enumerate(all_routes):
    ax.plot(routes['r1'][0], routes['r1'][1], color=colors_body1[idx], alpha=0.6)
    ax.plot(routes['r2'][0], routes['r2'][1], color=colors_body2[idx], alpha=0.6)
    ax.plot(routes['r3'][0], routes['r3'][1], color=colors_body3[idx], alpha=0.6)

for idx, routes in enumerate(all_routes):
    ax.scatter(routes['r1'][0, 0], routes['r1'][1, 0], color=colors_body1[idx], s=20, marker='o')
    ax.scatter(routes['r2'][0, 0], routes['r2'][1, 0], color=colors_body2[idx], s=20, marker='o')
    ax.scatter(routes['r3'][0, 0], routes['r3'][1, 0], color=colors_body3[idx], s=20, marker='o')

ax.set_title("Three-Body Problem - 10 Simulações")
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.legend(loc='upper right', fontsize='small')

# Salvar o gráfico como imagem
plt.savefig("three_body_routes.png")

# Exibir o gráfico
plt.show()

# Criar o gráfico base para a animação
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_title("Three-Body Problem - Trajetórias")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Extração das posições da primeira simulação para a animação
r1_base, r2_base, r3_base = all_routes[0]['r1'], all_routes[0]['r2'], all_routes[0]['r3']

# Linhas para as trajetórias
line1, = ax.plot([], [], 'b-', label="Body 1")
line2, = ax.plot([], [], 'g-', label="Body 2")
line3, = ax.plot([], [], 'r-', label="Body 3")

# Pontos para os corpos
point1, = ax.plot([], [], 'bo')
point2, = ax.plot([], [], 'go')
point3, = ax.plot([], [], 'ro')

ax.legend()

# Função de inicialização para a animação
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])
    point3.set_data([], [])
    return line1, line2, line3, point1, point2, point3

# Função de atualização para a animação
def update(frame):
    line1.set_data(r1_base[0, :frame], r1_base[1, :frame])
    line2.set_data(r2_base[0, :frame], r2_base[1, :frame])
    line3.set_data(r3_base[0, :frame], r3_base[1, :frame])
    
    point1.set_data([r1_base[0, frame]], [r1_base[1, frame]])
    point2.set_data([r2_base[0, frame]], [r2_base[1, frame]])
    point3.set_data([r3_base[0, frame]], [r3_base[1, frame]])
    
    return line1, line2, line3, point1, point2, point3

# Criar a animação
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, interval=20, blit=True)

# Salvar a animação como vídeo. Substitua "ffmpeg" por "pillow" se não tiver ffmpeg instalado
ani.save("three_body_simulation.gif", writer="pillow")

# Exibir o gráfico para validação
plt.show()
