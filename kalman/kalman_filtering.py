import numpy as np
from filterpy import kalman
from filterpy.common import Q_discrete_white_noise
from typing import Tuple, List


class VectorSizeMismatch(Exception):
    pass


class KalmanFiltering:
    def __init__(self, x: np.array,
                 dim_z: int,
                 time_step: float,
                 p_x: np.array,
                 r_x: np.array,
                 process_noise: float = 1e-3):
        """
        Фильтр Калмана как callable объект. Возвращает новую оценку вектора состояния по вызову __call__.
        :param x: Начальный вектор состояния.
        :param dim_z: Длина вектора наблюдений.
        :param p_x: Дисперсии начального вектора состояния.
        :param r_x: Шумы измерений.
        :param process_noise: Погрешность модели.
        :param time_step: шаг времени.
        """

        self.__x = x
        self.__p_x = p_x
        self.__r_x = r_x
        self.__process_noise = process_noise
        self.__dim_z = dim_z
        self.__time_step = time_step
        self.__filter = kalman.KalmanFilter(dim_x=len(self.__x),
                                            dim_z=self.__dim_z)
        # Хранилище векторов оценок состояния
        self.__filtered_state = list()
        # Хранилище квадратов ошибок оценки состояния
        self.__covariance_history = list()

    @property
    def filtered_state(self) -> List[np.array]:
        return self.__filtered_state

    @property
    def covariance_history(self) -> List[np.array]:
        return self.__covariance_history

    def setup(self) -> None:
        """
        Инициализация параметров фильтра Калмана.
        :return: None.
        """

        # Матрица процесса (ДУ движения)
        self.__filter.F = np.array([[1, 0, self.__time_step, 0, 0, 0, 0, 0],
                                    [0, 1, 0, self.__time_step, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, self.__time_step, 0],
                                    [0, 0, 0, 0, 0, 1, 0, self.__time_step],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]])
        # Матрица-связь вектора состояния с показаниями датчиков
        self.__filter.H = np.eye(len(self.__x))
        # Ковариационная матрица ошибки модели
        self.__filter.Q = Q_discrete_white_noise(dim=len(self.__x), dt=self.__time_step, var=self.__process_noise)
        # Ковариационная матрица ошибки измерений
        self.__filter.R = np.diag(self.__r_x)
        # Ковариационная матрица состояния
        self.__filter.P = np.diag(self.__p_x)

    def __call__(self, z: np.array, time_step: float) -> Tuple[np.array, np.array]:
        """
        Производит очерендую итерацию оценки вектора состояния по переданному вектору наблюдений.
        :param z: Вектор наблюдений.
        :param time_step: Шаг времени.
        :return: Кортеж вида (Последняя оценка вектора состояния, ковариационная матрица ошибок этой оценки).
        """

        if len(z) != self.__dim_z:
            raise VectorSizeMismatch(f"Передан вектор длиной {len(z)}, "
                                     f"ожидается {self.__dim_z}")

        self.__filter.Q = Q_discrete_white_noise(dim=len(self.__x), dt=time_step, var=self.__process_noise)

        self.__filter.predict()
        self.__filter.update(z)
        self.__filtered_state.append(self.__filter.x)
        self.__covariance_history.append(self.__filter.P)
        return self.__filtered_state[-1], self.__covariance_history[-1]
