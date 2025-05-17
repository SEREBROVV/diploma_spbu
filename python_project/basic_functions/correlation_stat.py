import basic_functions as bf
from scipy.stats import norm, chi2, t
import numpy as np
import math

def rang_for_spearman_method(x: list):
    # функция возвращает последовательность рангов для коэффициента корреляции Спирмена
    n = len(x)
    x_rang = []
    x.sort()
    indx = [1]
    x_step = x[0]
    for i in range(1, n):
        if x[i] == x_step:
            indx.append(i + 1)
        else:
            for _ in range(len(indx)):
                x_rang.append((x_step, sum(indx) / len(indx)))
            indx = [i + 1]
            x_step = x[i]
        if i == n - 1:
            for _ in range(len(indx)):
                x_rang.append((x_step, sum(indx) / len(indx)))

    return x_rang

def spearman_corr_coeff(x_y: list):
    # функция возвращает значение коэффициента корреляции Спирмена 

    # считаем размер выборки
    n = len(x_y)

    # разбиваем выборку из переменных на две выборки с одной переменной
    x = [x_y[i][0] for i in range(n)]
    y = [x_y[i][1] for i in range(n)]

    # считаем ранги для каждой выборки
    x_rang = rang_for_spearman_method(x)
    y_rang = rang_for_spearman_method(y)

    # считаем среднее значение рангов, т.к. в выборке не все значения уникальные
    x_rang_mean = (n + 1) / 2 # mean([i[1] for i in x_rang])
    y_rang_mean = (n + 1) / 2 # mean([i[1] for i in y_rang])

    # собираем список, каждый элемент которого имеет вид ((элемент первой выборки, его ранг), (элемент второй выборки, его ранг))
    x_y_rang = []
    for i in range(n):
        x_i_rang = 0
        for j in range(n):
            if x_y[i][0] == x_rang[j][0]:
                x_i_rang = x_rang[j][1]
                break

        y_i_rang = 0
        for j in range(n):
            if x_y[i][1] == y_rang[j][0]:
                y_i_rang = y_rang[j][1]
                break

        x_y_rang.append(((x_y[i][0], x_i_rang), (x_y[i][1], y_i_rang)))
    
    # считаем коэффициент корреляции Спирмена
    r_numerator = 0
    r_denominator_1 = 0
    r_denominator_2 = 0
    for i in range(n):
        r_numerator += (x_y_rang[i][0][1] - x_rang_mean) * (x_y_rang[i][1][1] - y_rang_mean)
        r_denominator_1 += (x_y_rang[i][0][1] - x_rang_mean) ** 2
        r_denominator_2 += (x_y_rang[i][1][1] - y_rang_mean) ** 2

    r_s = r_numerator / (((r_denominator_1) * (r_denominator_2)) ** (1 / 2))

    return r_s

def spearman_method(x_y: list, alpha, alternative='корреляция не равна 0'):
    # функция возвращает результат критерия значимости коэффициента корреляции Спирмена
    
    # считаем количество элементов в выборке
    n = len(x_y)
    
    # считаем коэффициент корреляции спирмена
    r_s = spearman_corr_coeff(x_y)

    # считаем статистику критерия
    s = r_s * (((n - 2) / (1 - r_s ** 2)) ** (1 / 2))

    # считаем p_value критерия
    if alternative == 'корреляция не равна 0':
        p_value = 2 * min(t.cdf(s, n - 2), 1 - t.cdf(s, n - 2))
        s_alpha = (float(t.ppf(alpha / 2, n - 2)), float(t.ppf(1 - (alpha) / 2, n - 2)))
    elif alternative == 'корреляция больше 0':
        p_value = 1 - t.cdf(s, n - 2)
        s_alpha = t.ppf(1 - alpha, n - 2)
    elif alternative == 'корреляция меньше 0':
        p_value = t.cdf(s, n - 2)
        s_alpha = t.ppf(alpha, n - 2)
    else:
        return "Неверно введена альтернативная гипотеза. Возможные варианты альтернативной гипотезы: 'корреляция не равна 0', 'корреляция больше 0', 'корреляция меньше 0'"
    
    ans = f'Корреляция равна {r_s} \nP-значение = {p_value} \nСтатистика критерия = {s} \nКритическое значение критерия = {s_alpha} \n'

    if p_value < alpha:
        ans += f'Поскольку p_value < alpha, то нулевая гипотеза об отсутствии корреляции между двумя выборками отвергается. Альтернативная гипотеза: {alternative}, не отвергается c уровнем значимости {alpha}'
    else:
        ans += f'Поскольку p_value > alpha, то нулевая гипотеза об отсутствии корреляции между двумя выборками не отвергается. Альтернативная гипотеза: {alternative}, отвергается c уровнем значимости {alpha}'
    
    return ans

def kendall_corr_coeff(x, y):
    # функция возвращает значение коэффициента корреляции Кендалла
    x_temp_set = set(x)  # отбираем уникальные значения из выборки x
    y_temp_set = set(y)  # отбираем уникальные значения из выборки y
    t_x_list = []  # список для хранения количества связей в x
    t_y_list = []  # список для хранения количества связей в y
    for i in x_temp_set:
        t = x.count(i)
        t_x_list.append(t * (t - 1) / 2)

    for i in y_temp_set:
        t = y.count(i)
        t_y_list.append(t * (t - 1) / 2)

    t_x = sum(t_x_list)  # считаем сумму количества связей в x
    t_y = sum(t_y_list)  # считаем сумму количества связей в x

    x_y = [(x[i], y[i]) for i in range(len(x))]
    x_y = sorted(x_y, key=lambda x: x[0])
    n = len(x_y)
    r_plus = 0  # для согласованных пар
    r_minus = 0  # для не согласованных пар
    
    # считаем согласованные и не соглаованные пары
    for i in range(n):
        for j in range(n):
            if x_y[j][0] == x_y[i][0] or x_y[j][1] == x_y[i][1]:
                continue
            elif (x_y[j][0] > x_y[i][0] and x_y[j][1] > x_y[i][1]):
                r_plus += 1
            elif (x_y[j][0] > x_y[i][0] and x_y[j][1] < x_y[i][1]):
                r_minus += 1
    
    # считаем коэффициент корреляции Кендалла
    tau_b = (r_plus - r_minus) / ((((n * (n - 1) / 2) - t_x) ** (1 / 2)) * (((n * (n - 1) / 2) - t_y) ** (1 / 2)))

    return tau_b 

def kendall_method(x, y, alpha, alternative = 'корреляция не равна 0'):
    # функция возвращает результат критерия значимости коэффициента корреляции Кендалла

    # считаем коэффициент корреляции Кендалла
    n = len(x)
    if len(y) != n:
        return 'Объемы выборок x и y должны совпадать'
    tau_b = kendall_corr_coeff(x, y)

    # считаем статистику критерия
    u_stat = tau_b * (((9 * n * (n - 1)) / (2 * (2 * n + 5))) ** (1 / 2))

    # вычисляем p_value и критическое значение критерия
    if alternative == 'корреляция не равна 0':
        p_value = 2 * min(norm.cdf(u_stat), 1 - norm.cdf(u_stat))
        u_alpha = (float(norm.ppf(alpha / 2)), float(norm.ppf(1 - (alpha) / 2)))
    elif alternative == 'корреляция больше 0':
        p_value = 1 - norm.cdf(u_stat)
        u_alpha = norm.ppf(1 - alpha)
    elif alternative == 'корреляция меньше 0':
        p_value = norm.cdf(u_stat)
        u_alpha = norm.ppf(alpha)
    else:
        return "Неверно введена альтернативная гипотеза. Возможные варианты альтернативной гипотезы: 'корреляция не равна 0', 'корреляция больше 0', 'корреляция меньше 0'"
    
    # формируем ответ
    ans = f'Корреляция равна {tau_b} \nP-значение = {p_value} \nСтатистика критерия = {u_stat} \nКритическое значение критерия = {u_alpha} \n'

    if p_value < alpha:
        ans += f'Поскольку p_value < alpha, то нулевая гипотеза об отсутствии корреляции между двумя выборками отвергается. Альтернативная гипотеза: {alternative}, не отвергается c уровнем значимости {alpha}'
    else:
        ans += f'Поскольку p_value > alpha, то нулевая гипотеза об отсутствии корреляции между двумя выборками не отвергается. Альтернативная гипотеза: {alternative}, отвергается c уровнем значимости {alpha}'
    
    return ans

def rang(x: list):
    # функция возвращает ранги для коэффициента конкордации

    n = len(x)
    x_rang = {}
    x_sorted = sorted(x)
    indx = [1]
    x_step = x_sorted[0]
    for i in range(1, n):
        if x_sorted[i] == x_step:
            indx.append(i + 1)
        else:
            x_rang[x_step] = sum(indx) / len(indx)
            indx = [i + 1]
            x_step = x_sorted[i]
        if i == n - 1:
            x_rang[x_step] = sum(indx) / len(indx)

    x_rang_new = []
    for i in x:
        x_rang_new.append(x_rang[i])

    return x_rang_new

def kendall_concord_coeff(x: list):
    # функция возвращает значение коэффициента конкордации

    m = len(x)
    n = len(x[0])
    x_rang = [rang(i) for i in x]
    x_rang_sum = m * (n * (n + 1) / 2)

    d = []
    for i in range(n):
        d.append(sum([j[i] for j in x_rang]) - x_rang_sum / n)
    d_2_sum = sum([i ** 2 for i in d])

    t = 0
    for i in x_rang:
        for j in set(i):
            cnt = i.count(j)
            if cnt != 1:
                t += (cnt ** 3 - cnt)

    w = 12 * d_2_sum / ((m ** 2) * (n ** 3 - n) - m * t)

    return w

def kendall_concord_method(x: list, alpha):
    # функция возвращает результат критерия значимости коэффициента конкордации

    m = len(x)
    n = len(x[0])
    x_rang = [rang(i) for i in x]
    x_rang_sum = m * (n * (n + 1) / 2)

    d = []
    for i in range(n):
        d.append(sum([j[i] for j in x_rang]) - x_rang_sum / n)
    d_2_sum = sum([i ** 2 for i in d])

    t = 0
    for i in x_rang:
        for j in set(i):
            cnt = i.count(j)
            if cnt != 1:
                t += (cnt ** 3 - cnt)

    w = 12 * d_2_sum / ((m ** 2) * (n ** 3 - n) - m * t)
    w_stat = 12 * d_2_sum / (m * n * (n + 1) - t / (n - 1))
    p_value = 1 - chi2.cdf(w_stat, n - 1)
    w_alpha = chi2.ppf(1 - alpha, n - 1)
    # формируем ответ
    ans = f'Конкордация равна {w} \nP-значение = {p_value} \nСтатистика критерия = {w_stat} \nКритическое значение критерия = {w_alpha} \n'

    if p_value < alpha:
        ans += f'Поскольку p_value < alpha, то нулевая гипотеза о равенстве нулю конкордации отвергается c уровнем значимости {alpha}.'
    else:
        ans += f'Поскольку p_value > alpha, то нулевая гипотеза о равенстве нулю конкордации не отвергается.'
    return ans