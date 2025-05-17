import basic_functions as bf
from scipy.stats import t, f
import numpy as np
    

class linear_regression:

    def __init__(self, y: list, X: list, x0_flg=True):

        self.y = np.array(y)
        self.n = len(y)

        if x0_flg:
            self.k = len(X)
            self.X = np.array([[1] * len(y)] + X).T
        else:
            self.k = len(X) - 1
            self.X = np.array(X).T

        self.coeff = np.array(np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.y))

        self.y_mean = bf.mean(self.y)
        self.r_2 = 1 - sum([(self.y[i] - sum(self.coeff * self.X[i])) ** 2 for i in range(self.n)]) / sum([(self.y[i] - self.y_mean) ** 2 for i in range(self.n)])

    def p_value(self):
        mse = sum([(self.y[i] - sum(self.coeff * self.X[i])) ** 2 for i in range(self.n)]) / (self.n - self.k - 1)
        var_b = mse * np.linalg.inv(np.dot(self.X.T, self.X)).diagonal()
        s_b = np.sqrt(var_b)
        t_stat = self.coeff / s_b
        p_value = [2 * min(t.cdf(i, self.n - self.k - 1), 1 - t.cdf(i, self.n - self.k - 1)) for i in t_stat]
        return p_value

    def loss(self):
        return [(self.y[i] - sum(self.coeff * self.X[i])) for i in range(self.n)]

    def loss_analyze(self):

        loss_result = self.loss()
        loss_mean = bf.mean(loss_result)

        if loss_mean > (10 ** (-4)) or loss_mean < (- 10 ** (-4)):
            return f'Математическое ожидаение оценок ненаблюдаемых случайных компонент равняется {loss_mean}, что протеворечит первой группе предположений регрессионного анализа'
        loss_corr = []
        for i in range(len(loss_result)):
            for j in range(len(loss_result)):
                if i != j:
                    loss_corr.append(loss_result[i] * loss_result[j])
        loss_corr_mean = bf.mean(loss_corr)
        if loss_corr_mean > (10 ** (-4)) or loss_corr_mean < (- 10 ** (-4)):
            return f'Оценки ненаблюдаемых случайных компонент являются линейно зависимыми, мат. ожидание = {loss_corr_mean}, что протеворечит первой группе предположений регрессионного анализа'
        ans = f'Предположения первой группы регрессионного анализа выполняются\n'
        
        r = bf.sturges(len(loss_result))
        loss_var = (bf.var(loss_result)) ** (1 / 2)
        alpha = 0.05
        res_chi = bf.chi2_method_for_norm(list(loss_result), alpha, 0, loss_var, r)

        if res_chi.split('\n')[1] == 'Гипотеза о нормальности распределия генеральной совокупности не отвегается':
            ans += f'Согласно критерию Пирсона, гипотеза о нормальности распределения ненаблюдаемых случайных компонент не отвергается с уровнем значимости {alpha}\n'
        
        res_kolmogorov = bf.kolmogorov_method_for_norm(list(loss_result), 0.05, 0, loss_var)
        if res_kolmogorov.split('\n')[1] == 'Гипотеза о нормальности распределия генеральной совокупности не отвегается':
            ans += f'Согласно критерию Колмогорова, гипотеза о нормальности распределения ненаблюдаемых случайных компонент не отвергается с уровнем значимости {alpha}\n'

        if ans == 'Предположения первой группы регрессионного анализа выполняются\n':
           ans += 'Предположения второй группый регрессионного анализа не выполняются'

        return ans

    def F(self, alpha=0.05):
        f_stat = ((self.r_2 ** 2) / (1 - self.r_2 ** 2)) * ((self.n - self.k - 1) / (self.k))
        p_value = 1 - f.cdf(f_stat, self.k, self.n - self.k - 1)
        f_crit = f.ppf(1 - alpha, self.k, self.n - self.k - 1)
        return f'Результат критерия Фишера о значимости построенной регрессионной модели\nalpha={alpha}\np_value = {p_value}\nСтатистика критерия = {f_stat}\nКритическое значение критерия = {f_crit}'