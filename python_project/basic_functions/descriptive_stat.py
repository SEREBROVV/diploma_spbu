import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean(x: list):
    # функция возвращает выборочное среднее выборки x
    n = len(x)
    m = sum(x) / n 
    return m 

def var(x: list):
    # функция возвращает выборочную дисперсию выборки x
    n = len(x)
    m = mean(x)
    s2 = (sum([(x[i] - m) ** 2 for i in range(n)])) / n 
    return s2 

def fix_var(x: list):
    # функция возвращает исправленную выборочную дисперсию выборки x
    n = len(x)
    s2 = var(x)
    s_2 = (n / (n - 1)) * s2
    return s_2 

def quartile(x: list):
    # функция возвращает список из квартилей выборки x
    n = len(x)
    x_sort = sorted(x)
    q1 = x_sort[(int(n * 0.25) + 1) - 1]  # индексы начинаются с 0
    q2 = x_sort[(int(n * 0.5) + 1) - 1]
    q3 = x_sort[(int(n * 0.75) + 1) - 1]
    return [q1, q2, q3]

def median(x: list):
    # функция возвращает медиану выборки
    n = len(x)
    x_sort = sorted(x)
    if n % 2 == 0:
        med = (x_sort[n // 2 - 1] + x_sort[(n // 2 + 1) - 1]) / 2
    else: 
        med = x[((n + 1) // 2) - 1]
    return med

def desc_stat(x: list, round_num = 2):
    # функция возвраащет словарь с описательной статистикой выборки x
    m = mean(x)
    s2 = var(x)
    s_2 = fix_var(x)
    q = quartile(x)
    med = median(x)
    min_x = min(x)
    max_x = max(x)
    ds = {'Выборочное среднее': m, 
          'Выборочная диспресия': s2,
          'Среднеквадратичное отклонение': s2 ** (1 / 2),
          'Исправленная выборочная дисперсия': s_2,
          'Минимальное значение': min_x,
          'Первый квартиль': q[0],
          'Второй квартиль': q[1],
          'Третий квартиль': q[2],
          'Максимальное значение': max_x,
          'Выборочная медиана': med
          }
    
    # округление
    if round_num != -1:
        for i in ds.keys():
            ds[i] = round(ds[i], round_num)
    
    tbl = {'Выборочная характеристика': list(ds.keys()), 'Значение': list(ds.values())}
    df = pd.DataFrame(tbl)

    return df

def freq(x: list):
    # функция возвращает словарь с частотами для каждого элемента выборки x
    n = len(x)
    y = []
    
    for i in range(n):
        if x[i] not in y:
            y.append(x[i])           
    y.sort()

    m = len(y)
    res = {}
    for i in range(m):
        res[y[i]] = sum([1 for j in range(n) if y[i] == x[j]])
        
    return res

def edf(x: list):
    # функция возвращает эмпирическая функция распределения выборки x
    n = len(x)
    y = freq(x)
    y_keys = list(y.keys())
    res = {}
    res[(float('-inf')), y_keys[0]] = 0
    for i in range(1, len(y_keys)):
        res[(y_keys[i - 1], y_keys[i])] = sum([y[y_keys[j]] for j in range(i)]) / n  
    res[(y_keys[-1], float('inf'))] = 1

    return res

def edf_val(x: dict, x0):
    # функция возвращает значение эмпирической функции распределения выборки x для аргумента x0
    for i in x.keys():
        if i[0] <= x0 < i[1]:
            ans = x[i]
            break
    return ans

def sturges(n: int):
    # функция возвращает значение формулы Стёрджесса
    k = int(1 + math.log2(n))
    return k

def h_for_kde(x: list):
    # функция возвращается длину интервала для ядерной оценки плотности
    k = sturges(len(x))
    h = round((max(x) - min(x)) / k, 2)
    return h

def kde(x: list):
    # функция возвращает список с аргументами и значениями ядерной оценки плотности
    y = np.linspace(min(x), max(x), 1000)
    h = h_for_kde(x)
    ans = []
    for i in range(len(y)):
        res = 0
        for j in range(len(x)):
            k_z = (1 / ((2 * math.pi) ** (1 / 2))) * math.exp((-(((y[i] - x[j]) / h)) ** 2) / 2)
            res += k_z
        res /= len(x) * h 
        ans.append(res)
    return [y, ans]

def triple_frequency_graph(x: list, attr_nm: str):
    # функция возвращает три графика: полигон частот, гистограмму частот и ядерную гистограмму
    fr = freq(x)
    k = sturges(len(x))
    kde_res = kde(x)
   
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.plot(fr.keys(), fr.values())
    plt.grid()
    #plt.title(f'Полигон частот  для {attr_nm}')

    plt.subplot(1, 3, 2)
    plt.hist(x, bins = k)
    plt.grid()
    #plt.title(f'Гистограмма частот  для {attr_nm}')

    plt.subplot(1, 3, 3)
    plt.plot(kde_res[0], kde_res[1])
    plt.grid()
    #plt.title(f'Ядерная гистограмма  для {attr_nm}')

    plt.show()


