import numpy as np
from scipy import special


def kurant_cond(cond_value, Nx, T):
    min_nt = Nx
    tau = T / (min_nt - 1)

    while tau >= cond_value:
        min_nt += 1
        tau = T / (min_nt - 1)

    return min_nt


def u_an_1(x, t, a, delta):
    u = 0

    for k in range(40):
        # Вычисление очередного слагаемого
        term = 1 / (2 * k + 1) ** 3 * np.exp(-(a * np.pi * (2 * k + 1)) ** 2 * t) * np.sin(np.pi * (2 * k + 1) * x)
        # Добавление слагаемого к сумме
        u += term

    return u


def u_an_2(x, t, a, delta):
    u = (np.exp(-(a * np.pi) ** 2 * t) * np.sin(np.pi * x) +
         1 / 2 * np.exp(-(3 * a * np.pi) ** 2 * t) * np.sin(3 * np.pi * x))

    return u


def u_an_3(x, t, a, delta):
    u = 0

    for k in range(1, 40):
        if k % 4 != 0:
            c_n = 4 / (np.pi * k) * np.sin(np.pi * k / 4) * np.sin(np.pi * k / 2 * delta)
        else:
            c_n = 0

        term = c_n * np.exp(-(a * np.pi * k) ** 2 * t) * np.sin(np.pi * k * x)

        u += term

    return u


def u_an_4(x, t, a, delta):
    u = 1 / np.sqrt(4 * a ** 2 * t + 1) * np.exp(-(x - 0.5) ** 2 / (4 * a ** 2 * t + 1))

    return u


def u_an_5(x, t, a, delta):
    u = 1 / 2 * (special.erf((3 / 4 - x) / (2 * np.sqrt(a ** 2 * t))) - special.erf(
        (1 / 4 - x) / (2 * np.sqrt(a ** 2 * t))))

    return u


def norm_estimate(u, h):
    # Получение размеров массива u
    Nx, Nt = u.shape

    # Создание пустого массива для хранения значений нормы оценки решения
    n = np.zeros(Nt)

    # Цикл по времени
    for j in range(Nt):
        # Вычисление нормы оценки решения по формуле
        n[j] = np.sqrt(h * np.sum(u[1:-1, j] ** 2))

    # Возврат результата
    return n


def solve_heat_equation(a, phi, psi_1, psi_2, f, L, T, Nx, Nt, cauchy, phi_num):
    """
    Функция для решения одномерного уравнения теплопроводности
    u'_t = a^2 * u''_xx + f(x, t) на отрезке [0, L] по x и на интервале [0, T] по t
    с начальным условием u(x, 0) = phi(x) и граничными условиями
    u(0, t) = psi_1(t), u(L, t) = psi_2(t)

    Параметры:
    a - коэффициент теплопроводности
    phi - функция для начального условия
    psi_1 - функция для левого граничного условия
    psi_2 - функция для правого граничного условия
    f - функция для источника тепла
    L - правая граница по x
    T - правая граница по t
    Nx - количество узлов по x
    Nt - количество узлов по t

    Возвращает:
    u - двумерный массив значений температуры в узлах сетки размером Nx * Nt
    """

    # Шаги сетки по x и по t
    if cauchy:
        if phi_num == 4:
            h = (2 * L) / (Nx - 1)
        else:
            h = L / (Nx - 1)
    else:
        h = L / (Nx - 1)

    tau = T / (Nt - 1)

    # Массивы значений x и t
    if cauchy:
        if phi_num == 4:
            x = np.linspace(-L, L, Nx)
        else:
            x = np.linspace(0, L, Nx)
    else:
        x = np.linspace(0, L, Nx)

    t = np.linspace(0, T, Nt)

    # Двумерный массив для хранения значений u на сетке
    u = np.zeros((Nx, Nt))

    # Задание начального условия
    u[:, 0] = phi(x)

    # Задание граничных условий
    u[0, :] = psi_1(t)
    u[-1, :] = psi_2(t)

    # Константа для явной разностной схемы
    c = a ** 2 * tau / h ** 2

    # Цикл по времени
    for j in range(Nt - 1):
        # Цикл по пространству
        for i in range(1, Nx - 1):
            # Явная разностная схема
            u[i, j + 1] = u[i, j] + c * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) + tau * f(x[i], t[j])

    # Возврат результата
    return u
