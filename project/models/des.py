from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import mean_squared_error, calinski_harabasz_score
import operator
import matplotlib.pyplot as plt
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DES_PALR(object):
    serie = None
    original_serie = None
    windows = None
    X_train = None
    y_train = None
    x_test = None
    y_test = None
    pool = None
    C = None
    U = None
    window_size = 5
    precision_by_cluster = []

    def __init__(self, serie, Z=10):
        self.serie = serie
        self.Z = Z
        self._build()

    def _build(self):
        self._normalize()
        self._generate_windows()
        self._split_data()
        self._generate_pool()
        self._calc_clusters()

    def _rebuild(self, count_new_samples=1, plot=True):
        self._normalize(plot=plot)
        self._generate_windows()
        self._split_data(new_index=count_new_samples)
        self._generate_pool()
        self._calc_clusters(plot=plot)

    def _normalize(self, plot=False):
        minimo = min(self.serie)
        maximo = max(self.serie)
        self.original_serie = self.serie
        self.serie = (self.serie - minimo) / (maximo - minimo)

        if plot:
            plt.plot(self.serie)
            plt.title('Normalized Serie')
            plt.xlabel('Time')
            plt.show()

    def _denormalize(self, s):
        minimo = min(serie_real)
        maximo = max(serie_real)
        serie = (serie_atual * (maximo - minimo)) + minimo
        return pd.DataFrame(serie)

    def _generate_windows(self):
        # serie: vetor do tipo numpy ou lista
        tam_janela = self.window_size
        serie = self.serie
        tam_serie = len(serie)
        tam_janela = tam_janela + 1  # Adicionado mais um ponto para retornar o target na janela

        janela = list(serie[0:0 + tam_janela])  # primeira janela p criar o objeto np
        janelas_np = np.array(np.transpose(janela))

        for i in range(1, tam_serie - tam_janela):
            janela = list(serie[i:i + tam_janela])
            j_np = np.array(np.transpose(janela))
            janelas_np = np.vstack((janelas_np, j_np))

        self.windows = janelas_np

    def _split_data(self, perc_train=0.7, new_index=0):
        # faz corte na serie com as janelas já formadas
        windows = self.windows
        x_date = windows[:, 0:-1]
        y_date = windows[:, -1]
        train_size = np.fix(len(windows) * perc_train)
        train_size = train_size.astype(int)

        self.X_train = x_date[0:train_size + new_index, :]
        self.y_train = y_date[0:train_size + new_index]
        self.x_test = x_date[train_size + new_index:-1, :]
        self.y_test = y_date[train_size + new_index:-1]

        logger.debug("Train partition: %d %d", 0, train_size + new_index)
        logger.debug("Test partition: %d %d", train_size + new_index, len(y_date))

    def _generate_pool(self):
        f_activation = ['sigmoid', 'sine', 'hardlim', 'tribas', 'gaussian']
        n_hidden = [i for i in range(1, 21)]

        pool = []
        for activation in f_activation:
            for n_neurones in n_hidden:
                try:
                    model = ELMRegressor(n_hidden=n_neurones, activation_func=activation)
                    model.fit(self.X_train, self.y_train)
                    pool.append(model)
                except Exception as e:
                    logging.debug(activation, n_neurones, ' - ', e)
        logging.debug('Pool size: %d', len(pool))
        self.pool = pool

    def _calc_clusters(self, plot=False):
        best_kmeans = None
        best_vrc = np.Inf
        results = []
        size = len(self.X_train)
        logger.debug('size: %d', size)
        max_k = int(np.sqrt(size))
        logger.debug('max_k: %d', max_k)
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(self.X_train)
            vrc = calinski_harabasz_score(self.X_train, kmeans.labels_)
            results.append([k, vrc])
            if vrc < best_vrc:
                best_vrc = vrc
                best_kmeans = kmeans
        logger.debug('Best K: %d', best_kmeans.n_clusters)
        self.C, self.U = best_kmeans.labels_, best_kmeans.cluster_centers_

        if plot:
            results = np.array(results)
            plt.plot(results[:, 0], results[:, 1], label='Score')
            plt.title('Calinski-Harabasz Index')
            plt.xlabel('K')
            plt.legend()
            plt.show()

            df = pd.DataFrame(best_kmeans.labels_)
            df[0].value_counts().plot.bar()
            plt.title('Count train exemples by clusters')
            plt.xlabel('Cluster')
            plt.ylabel('Number of exemples')
            plt.show()

    def _select_predictors(self, x_val, y_val):
        ensemble = {}
        for model in self.pool:
            pred = model.predict(x_val)
            error = np.sqrt(mean_squared_error(y_val, pred))
            ensemble[model] = error

        ensemble = [m for m, e in sorted(
            ensemble.items(), key=operator.itemgetter(1))][:self.Z]
        return ensemble

    def _forecast(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        C_i = np.argmin(distances)
        logger.debug('Ci = %d', C_i)

        selected_indexs = (self.C == self.C[C_i])
        x_val = self.X_train[selected_indexs]
        y_val = self.y_train[selected_indexs]

        ensemble = self._select_predictors(x_val, y_val)
        predictions = []
        logger.debug('Number of selected predictors: %d', len(ensemble))
        for model in ensemble:
            p = model.predict(x)
            predictions.append(p)

        return np.mean(predictions, axis=0), self.C[C_i]

    def forecast(self, x):
        return self._forecast(x)[0]

    def test(self, auto=False, show_results=True, plot=False):
        predictions = []
        x_test = np.copy(self.x_test)
        y_test = np.copy(self.y_test)
        for i in range(len(x_test)):
            x = x_test[i]
            y = y_test[i]
            p, cluster = self._forecast(np.array([x]))
            predictions.append(p)
            error = mean_squared_error([y], p)
            self.precision_by_cluster.append([cluster, error])

            if (auto):
                self._rebuild(i + 1, plot=plot)

        if plot:
            pd.DataFrame(ensemble.precision_by_cluster).boxplot(by=0)
            plt.title('Boxplot MSE by Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('MSE')
            plt.show()

        if show_results:
            plt.plot(y_test, label='Test')
            plt.plot(predictions, label='Forecast')
            plt.title('Test Forecast')
            plt.xlabel('Time')
            plt.legend()
            plt.show()

        return mean_squared_error(y_test, predictions)
