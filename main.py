import sys

import numpy as np
import matplotlib.pyplot as plt

from user_interaction import user_interface
from utils import solve_heat_equation, norm_estimate, u_an_1, u_an_2, u_an_3, u_an_4, u_an_5, kurant_cond

# Параметры задачи
params = user_interface()
an_func = [u_an_1, u_an_2, u_an_3, u_an_4, u_an_5]
a = params['a']
L = params['L']  # правая граница по x
T = 1  # правая граница по t
Nx = params['Nx']  # количество узлов по x

if params['cauchy']:
    if params['phi_num'] == 4:
        h = (2 * L) / (Nx - 1)
    else:
        h = L / (Nx - 1)
else:
    h = L / (Nx - 1)

cond_value = h ** 2 / (2 * a ** 2)  # условие Куранта

Nt = kurant_cond(cond_value, Nx, T)  # количество узлов по t
print(f'Nt = {Nt}')

delta = params['delta']
phi = 0
match params['phi_num']:
    case 1:
        phi = lambda x: 4 * (1 - x) * x
    case 2:
        phi = lambda x: np.sin(np.pi * x) + 1 / 2 * np.sin(3 * np.pi * x)
    case 3:
        phi = lambda x: np.where(np.abs(x - 0.5) <= delta, 1, 0)
    case 4:
        phi = lambda x: np.exp(-(x - 0.5) ** 2)
    case 5:
        phi = lambda x: np.where(np.abs(x - 0.5) <= 1 / 4, 1, 0)

if not params['cauchy']:
    psi_1 = lambda t: 0  # функция для левого граничного условия
    psi_2 = lambda t: 0  # функция для правого граничного условия
else:
    if params['phi_num'] == 4:
        psi_1 = lambda t: an_func[params['phi_num'] - 1](-L, t, a, delta)
        psi_2 = lambda t: an_func[params['phi_num'] - 1](L, t, a, delta)
    else:
        psi_1 = lambda t: an_func[params['phi_num'] - 1](0, t, a, delta)
        psi_2 = lambda t: an_func[params['phi_num'] - 1](1, t, a, delta)

f = lambda x, t: 0  # функция для источника тепла

# Решение уравнения теплопроводности с помощью явной разностной схемы
try:
    u_num = solve_heat_equation(a, phi, psi_1, psi_2, f, L, T, Nx, Nt, params['cauchy'], params['phi_num'])
except ValueError:
    sys.exit()

# Массивы значений x и t
if params['cauchy']:
    if params['phi_num'] == 4:
        x = np.linspace(-L, L, Nx)
    else:
        x = np.linspace(0, L, Nx)
else:
    x = np.linspace(0, L, Nx)

t = np.linspace(0, T, Nt)

# Нахождение аналитического решения на сетке
u_an = np.zeros((Nx, Nt))
for j in range(1, Nt):
    for i in range(Nx):
        u_an[i, j] = an_func[params['phi_num'] - 1](x[i], t[j], a, delta)

# Подсчет нормы оценки решения
n = norm_estimate(u_num - u_an, h)

# Выбор моментов времени для построения графиков температуры
# Можно выбрать любые значения из массива t или использовать индексы
t_values = [0, T]  # значения времени
t_indices = [0, -1]  # индексы соответствующих значений времени

# Создание фигуры с одним подграфиком
fig, ax = plt.subplots(figsize=(8, 4))

# Построение графиков температуры в зависимости от x для всех трех решений
ax.plot(x, phi(x), color='green', marker='s', label=f't = 0')  # начальное решение
ax.plot(x, u_num[:, t_indices[1]], color='blue', marker='o',
        label=f't = {T}, n={Nx - 1}, Err={n[t_indices[1]]:.3e}')  # численное решение
ax.plot(x, u_an[:, t_indices[1]], color='red', marker='x', linestyle='--',
        label=f'u_an(x), t = {T}')  # аналитическое решение

# Добавление заголовка и подписей осей для подграфика
ax.set_title('Решение одномерного уравнения теплопроводности')
ax.set_xlabel('x')
ax.set_ylabel('t')
# Добавление сетки для улучшения читаемости графика
ax.grid()
# Добавление легенды для обозначения разных решений и показателей
ax.legend()

# Добавление общего заголовка для всей фигуры
fig.suptitle('Сравнение трех решений одномерного уравнения теплопроводности')

# Показать фигуру на экране
plt.show()
