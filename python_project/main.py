import basic_functions as bf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from itertools import combinations


st_ls = pd.read_csv('student_lifestyle_dataset.csv')

st_ls = st_ls.rename(columns={'Student_ID': 'id',
                            'Study_Hours_Per_Day': 'study_hour_cnt',
                            'Extracurricular_Hours_Per_Day': 'extra_study_hour_cnt',
                            'Sleep_Hours_Per_Day': 'sleep_hour_cnt',
                            'Social_Hours_Per_Day': 'social_hour_cnt',
                            'Physical_Activity_Hours_Per_Day': 'sport_hour_cnt',
                            'GPA': 'gpa',
                            'Stress_Level': 'stress_level',
                            })



# описательная статистика
# x = list(st_ls['gpa'])
# ds = bf.desc_stat(x)
# print(ds)

# полигон частот
# x = list(st_ls['gpa'])
# fr = bf.freq(x)
# plt.plot(fr.keys(), fr.values())
# plt.show()

# гистограмма
# x = list(st_ls['gpa'])
# k = bf.sturges(len(x))
# plt.hist(x, bins = k)
# plt.show()

# ядерная гистограмма
# x = list(st_ls['gpa'])
# kde_res = bf.kde(x)
# plt.plot(kde_res[0], kde_res[1])
# plt.grid()
# plt.show()

# три графика (полигон частот, гистограмма частот, ядерная гистограмма) в одном
# attr_nm = 'sport_hour_cnt'
# x = list(st_ls[attr_nm])
# bf.triple_frequency_graph(x, attr_nm)

# критерий согласия Пирсона хи2 с гипотезой о нормальности распределения генеральной совокупности
# x = list(st_ls['gpa'])
# print(bf.chi2_method_for_norm(x = x, alpha = 0.05, x_mean = bf.mean(x), x_var = (bf.var(x)) ** (1 / 2), r = 20))

# критерий согласия Колмогорова о нормальности распределения генеральной совокупности
# pre_x = st_ls['sport_hour_cnt']
# x = [math.log(i) for i in pre_x if i != 0]
# print(bf.kolmogorov_method_for_norm(x = x, alpha = 0.05, x_mean = bf.mean(x), x_var = (bf.var(x)) ** (1 / 2)))

# критерий однородности Колмогорова-Смирнова
# x = list(st_ls['study_hour_cnt'])
# y = list(st_ls['sleep_hour_cnt'])
# print(bf.kolmogorov_smirnov_method(x, y, 0.05))

# корреляционный анализ
# изобразим две выборки на одном графике
# x_name = 'study_hour_cnt'
# y_name = 'gpa'
# x = list(st_ls[x_name])
# y = list(st_ls[y_name])
# plt.plot(x, y, 'o')
# plt.xlabel(x_name)
# plt.ylabel(y_name)
# plt.show()

# находим корреляцию с помощью критерия Спирмена
# x = list(st_ls['sport_hour_cnt'])
# y = list(st_ls['social_hour_cnt'])
# x_y = [(x[i], y[i]) for i in range(len(x))]
# print(bf.spearman_method(x_y, alpha=0.05, alternative='корреляция меньше 0'))

# строим тепловую карту с коэффициентами корреляции Спирмена
# sp_coeff = []
# col = list(st_ls.columns[1:len(st_ls.columns) - 1]) # не рассматриваем id и stress_level
# for i in range(len(col)):
#     sp_coeff.append([])
#     for j in range(len(col)):
#         x_y = [(st_ls[col[i]][k], st_ls[col[j]][k]) for k in range(len(st_ls[col[i]]))]
#         sp_coeff[i].append(bf.spearman_corr_coeff(x_y))

# sp_coeff_df = pd.DataFrame(sp_coeff, index = col, columns = col)
# sns.heatmap(sp_coeff_df, vmin = -1.0, vmax = 1.0, cmap = 'seismic', center = 0.0, annot = True, linewidth=.5)
# plt.show()

# находим корреляцию с помощью критерия Кендалла
# x = list(st_ls['study_hour_cnt'])
# y = list(st_ls['gpa'])
# alpha = 0.05
# print(bf.kendall_method(x, y, alpha = 0.05, alternative='корреляция больше 0'))

# строим тепловую карту с коэффициентами корреляции Кендалла
# kl_coeff = []
# col = list(st_ls.columns[1:len(st_ls.columns) - 1])   # не рассматриваем id и stress_level
# for i in range(len(col)):
#     x = list(st_ls[col[i]])
#     kl_coeff.append([])
#     for j in range(len(col)):
#         y = list(st_ls[col[j]])
#         kl_coeff[i].append(bf.kendall_corr_coeff(x, y))

# kl_coeff_df = pd.DataFrame(kl_coeff, index = col, columns = col)
# sns.heatmap(kl_coeff_df, vmin = -1.0, vmax = 1.0, cmap = 'seismic', center = 0.0, annot = True, linewidth=.5) # cmap = 'YlOrBr'
# plt.show()

# находим конкордацию для нескольких выборок
# col = ['social_hour_cnt', 'sport_hour_cnt'] # , 'sleep_hour_cnt', 'extra_study_hour_cnt', 'social_hour_cnt', 'sport_hour_cnt'
# for i in col:
#     x = [list(st_ls['gpa']), list(st_ls['study_hour_cnt']), list(st_ls['sleep_hour_cnt']), list(st_ls['extra_study_hour_cnt']), list(st_ls[i])]
#     print(f'gpa, study_hour_cnt, sleep_hour_cnt, {i} - {bf.kendall_concord_coeff(x)}')

# print(bf.kendall_concord_method(x, 0.05))

# регрессионный анализ
# y_name = 'gpa'
# x_names = ['study_hour_cnt', 'extra_study_hour_cnt', 'sleep_hour_cnt', 'social_hour_cnt', 'sport_hour_cnt']

# all_x_names = []
# for i in range(len(x_names)):
#     for j in list(combinations(x_names, i + 1)):
#         all_x_names.append(list(j))

# y = list(st_ls[y_name])
# for x_list_name in all_x_names:
#     X = []
#     if 'study_hour_cnt' in x_list_name:
#         for x_name in x_list_name:
#             X.append(np.array(st_ls[x_name]))
#         try:
#             lm = bf.linear_regression(y, X)
#             coeff = lm.coeff
#             if coeff is not None:
#                 r_2 = lm.r_2
#                 p_value = lm.p_value()
#                 res = {'Название атрибута': ['свободный член'] + x_list_name, 'Коэффициент': coeff, 'p_value': p_value}
#                 res_df = pd.DataFrame(res)
#                 print(res_df)
#                 print(r_2)
#                 print(lm.loss_analyze())
#                 print(lm.F())
#                 print('----------------------------------------------------')
#         except:
#             print('Матрица X является вырожденной, коэффициенты линейной регрессии не могут быть рассчитаны с помощью МНК')

