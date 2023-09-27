import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import solve_heat_equation, norm_estimate, u_an_1, u_an_2, u_an_3, u_an_4

# Параметры задачи
a = 0.3  # коэффициент теплопроводности
delta = 0.125
cauchy = False

"""phi = lambda x: np.sin(np.pi * x) + 1 / 2 * np.sin(3 * np.pi * x)  # функция для начального условия"""
phi = lambda x: np.where(np.abs(x - 0.5) <= delta, 1, 0)
"""phi = lambda x: 4 * (1 - x) * x  # функция для начального условия"""
"""phi = lambda x: np.exp(-(x - 0.5) ** 2)"""
psi_1 = lambda t: 0  # функция для левого граничного условия
psi_2 = lambda t: 0  # функция для правого граничного условия
f = lambda x, t: 0  # функция для источника тепла

L = 1  # правая граница по x
T = 1  # правая граница по t
Nx = 65  # количество узлов по x
Nt = 739  # количество узлов по t

# Вычисление границ для задачи Коши
"""psi_1 = lambda t: u_an_4(-L, t, a)
psi_2 = lambda t: u_an_4(L, t, a)"""

# Решение уравнения теплопроводности с помощью явной разностной схемы
try:
    u_num = solve_heat_equation(a, phi, psi_1, psi_2, f, L, T, Nx, Nt, cauchy)
except ValueError:
    sys.exit()

# Массивы значений x и t
if cauchy:
    x = np.linspace(-L, L, Nx)
else:
    x = np.linspace(0, L, Nx)

t = np.linspace(0, T, Nt)

# Шаг сетки по x
if cauchy:
    h = (2 * L) / (Nx - 1)
else:
    h = L / (Nx - 1)

# Нахождение аналитического решения на сетке
u_an = np.zeros((Nx, Nt))
for j in range(Nt):
    for i in range(Nx):
        u_an[i, j] = u_an_3(x[i], t[j], a, delta)

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
