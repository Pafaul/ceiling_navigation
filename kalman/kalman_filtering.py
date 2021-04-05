import numpy as np


class KalmanFiltering:
    def __init__(self, x: np.array,
                 z: np.array,
                 r: np.array,
                 en: np.array,
                 d_noise: np.array):
        """
        Фильтрация Калмана как функция с состоянием.
        Каждая следующая итерация происходит по вызову callable объекта.

        :param x: Начальный вектор состояния
        :param z: Начальный вектор наблюдений
        :param r: Вектор корреляций между координатами вектора состояний
        :param en: Вектор возмущений (возмущения нормальные: M = 0; дисперсии координат некоррелированы)
        :param d_noise: Вектор ошибок наблюдений (ошибки нормальные: M = 0; дисперсии координат некоррелированы)
        """
        self.__x = x
        self.__z = z
        self.__r = np.diag(r)
        self.__v_ksi = np.diag(en)
        self.__v = np.diag(d_noise)
        self.__v_inv = np.linalg.inv(self.__v)
        self.__observation_number = 1
        # формирование оценки перемещения и вектора дисперсий ошибок этой оценки на первый шаг
        self.__xx, self.__p = self.__setup()

    def __setup(self):
        # вектор оценок перемещения
        xx = np.zeros(self.__observation_number * len(self.__x)) \
            .reshape(self.__observation_number, len(self.__x))
        # вектор дисперсий ошибок оценивания
        p = self.__v
        # первая оценка
        xx[:][0] = self.__z
        return xx, p

    def __call__(self, z: np.array, d_noise: np.array, **kwargs):
        self.__v = np.diag(d_noise)
        self.__v_inv = np.linalg.inv(self.__v)
        pe = np.dot(np.dot(self.__r, self.__p), self.__r.T) + self.__v_ksi
        self.__p = np.dot(pe, self.__v) * np.linalg.inv(pe + self.__v)
        xe = np.dot(self.__r, self.__xx[:][self.__observation_number - 1])
        iteration_result = self.__xx[:][self.__observation_number] = xe + np.dot(np.dot(self.__p, self.__v_inv),
                                                                                 (z - xe))
        self.__observation_number += 1
        return self.__p, iteration_result
