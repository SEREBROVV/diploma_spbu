import basic_functions as bf
from scipy.stats import norm, chi2, kstwobign
import numpy as np
import math

def chi2_method_for_norm(x: list, alpha, x_mean, x_var, r):
    # функция возвращает результат критерия согласия хи квадрат Пирсона для простой гипотезы о подчинении нормальному закону распределения

    # делим числовую ось на r промежутков
    n = len(x)
    y = []
    y.append(float('-inf'))
    for i in np.linspace(min(x), max(x), r):
        y.append(round(float(i), 2))
    y[-1] = float('inf')
    
    # считаем частоты в каждом интервале
    # частота в каждом интервале должна быть больше 5
    # если частота в интервале меньше 5, то мы его схлопываем со следующим интервалом и считаем частоту уже в "двойном" интервале
    i = 0
    cnt = 0
    step = 0
    res = {}
    while i + step < r:
        if cnt < 5:
            step += 1
        else:
            res[(y[i], y[i + step])] = [cnt]
            i += step
            step = 1
        cnt = 0
        for j in range(n):
            if y[i] < x[j] <= y[i + step]:
                cnt += 1
        if i + step == r:
            if cnt < 5:
                intrvl = list(res.keys())[-1]
                val = res[intrvl][0]
                del res[intrvl]
                res[(intrvl[0], y[i + step])] = [val + cnt]
            else:
                res[(y[i], y[i + step])] = [cnt]
            break
    
    # если в результате получилось меньше 4 интервалов, значит критерий согласия хи2 Пирсона не подходит для данной выборки
    if len(res) < 4:
       ans = 'Критерий согласия хи2 Пирсона не может быть применен к данной выборки'
       return ans
    
    # считаем вероятности для каждого интервала с помощью функции нормального распределения
    for i in res.keys():
        res[i].append(norm.cdf(i[1], loc = x_mean, scale = x_var) - norm.cdf(i[0], loc = x_mean, scale = x_var))
    
    # считаем статистику критерия, p-значение и критическое значение критерия
    chi_stat = sum([((res[i][0] - n * res[i][1]) ** (2)) / (n * res[i][1])  for i in res.keys()])
    p_value = 1 - chi2.cdf(chi_stat, len(res) - 1)
    chi_alpha = chi2.ppf(1 - alpha, len(res) - 1)
    
    # результат критерия
    ans = 'Результаты критерия согласия хи2 Пирсона: \n'
    if p_value > alpha:
        ans += f'Гипотеза о нормальности распределия генеральной совокупности не отвегается\n'
    else:
        ans += f'Гипотеза о нормальности распределия генеральной совокупности отвегается c уровнем значимости {alpha}\n'
    ans += f'Статистика критерия: {chi_stat}\nКритическое значение критерия: {chi_alpha}\np-значение: {p_value}'

    return ans

def kolmogorov_method_for_norm(x: list, alpha, x_mean, x_var):
    # функция возвращает результат критерия согласия Колмогорова для простой гипотезы о подчинении нормальному закону распределения

    x.sort()
    n = len(x)

    # считаем значение статистики Колмогорова
    d_n_plus = max([(i / n) - norm.cdf(x[i - 1], x_mean, x_var) for i in range(1, n + 1)])
    d_n_minus = max([norm.cdf(x[i - 1], x_mean, x_var) - ((i - 1) / n) for i in range(1, n + 1)])

    d_n = max([d_n_plus, d_n_minus])

    k_stat = (n ** (1 / 2)) * d_n
    p_value = 1 - kstwobign.cdf(k_stat)
    k_alpha = kstwobign.ppf(1 - alpha)
    
    ans = 'Результаты критерия согласия Колмогорова: \n'
    if p_value > alpha:
        ans += f'Гипотеза о нормальности распределия генеральной совокупности не отвегается\n'
    else:
        ans += f'Гипотеза о нормальности распределия генеральной совокупности отвегается c уровнем значимости {alpha}\n'
    ans += f'Статистика критерия: {k_stat}\nКритическое значение критерия: {k_alpha}\np-значение: {p_value}'
    
    return ans

def kolmogorov_smirnov_method(x: list, y: list, alpha=0.05):
    # функция возвращает результат критерия однородности Колмогорова-Смирнова

    # вычисляем длину двух выборок
    n = len(x)
    m = len(y)
    
    # вычисляем эмпирические функции распределения двух выборок
    x_edf = bf.edf(x)
    y_edf = bf.edf(y)

    # вычисляем статистику критерия
    d_m_n = 0 
    x_y = x + y  # объединяем выборки для вычисления статистики критерия
    for i in range(len(x_y)):
        val = abs(bf.edf_val(x_edf, x_y[i]) - bf.edf_val(y_edf, x_y[i]))  # считаем разность между значениями эмпирических функций распределений
        if val > d_m_n:
            d_m_n = val
    
    d_stat = d_m_n * (((n * m) / (n + m)) ** (1 / 2))
    
    # вычисляем p_value и критическое значение критерия
    d_alpha = kstwobign.ppf(1 - alpha)
    p_value = 1 - kstwobign.cdf(d_stat)

    # формурием ответ
    ans = 'Результаты критерия однородности Колмогорова-Смирнова: \n'
    if p_value > alpha:
        ans += f'Гипотеза о равенстве функций распределений двух генеральных совокупностей не отвегается\n'
    else:
        ans += f'Гипотеза о равенстве функций распределений двух генеральных совокупностей отвегается c уровнем значимости {alpha}\n'
    ans += f'Статистика критерия: {d_stat}\nКритическое значение критерия: {d_alpha}\np-значение: {p_value}'
    
    return ans