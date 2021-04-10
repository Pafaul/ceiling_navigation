import numpy as np
from filterpy import kalman
from filterpy.common import Q_discrete_white_noise
from typing import Tuple, List
import matplotlib.pyplot as plt


class VectorSizeMismatch(Exception):
    pass


class KalmanFiltering:
    def __init__(self,
                 x: np.array,
                 dim_z: int,
                 p_x: np.array,
                 r_x: np.array,
                 process_noise: float = 1.):
        """
        Фильтр Калмана как callable объект. Возвращает новую оценку вектора состояния по вызову __call__.
        :param x: Начальный вектор состояния.
        :param dim_z: Длина вектора наблюдений.
        :param p_x: Дисперсии начального вектора состояния.
        :param r_x: Шумы измерений.
        :param process_noise: Погрешность модели.
        """

        self.__x = x
        self.__p_x = p_x
        self.__r_x = r_x
        self.__process_noise = process_noise
        self.__dim_z = dim_z
        self.__filter = kalman.KalmanFilter(dim_x=len(self.__x),
                                            dim_z=self.__dim_z)

        self.__descriptions = [
            "X",
            "Y",
            "Z",
            "VX",
            "VY",
            "VZ",
            "alpha",
            "beta",
            "gamma",
            "wx",
            "wy",
            "wz"
        ]
        # Хранилище векторов оценок состояния
        self.__filtered_state = list()
        # Хранилище квадратов ошибок оценки состояния
        self.__covariance_history = list()
        self.__z_history = list()
        self.__setup()

    @property
    def filtered_state(self) -> List[np.array]:
        return np.array(self.__filtered_state)

    @property
    def covariance_history(self) -> List[np.array]:
        return np.array(self.__covariance_history)

    @property
    def z_history(self):
        return np.array(self.__z_history)

    def reset(self, x: np.array):
        if len(self.__x) != x:
            raise VectorSizeMismatch(f"Передан вектор длиной {len(x)}, "
                                     f"ожидается {len(self.__x)}")
        self.__x = x
        self.__filtered_state.clear()
        self.__covariance_history.clear()
        self.__setup()

    def __setup(self) -> None:
        """
        Инициализация параметров фильтра Калмана.
        :return: None.
        """
        # Матрица процесса (ДУ движения)
        # self.__filter.F = np.array([[1, 0, 0, self.__time_step, 0, 0, 0, 0, 0, 0, 0, 0],
        #                             [0, 1, 0, 0, self.__time_step, 0, 0, 0, 0, 0, 0, 0],
        #                             [0, 0, 1, 0, 0, self.__time_step, 0, 0, 0, 0, 0, 0],
        #                             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #                             [0, 0, 0, 0, 0, 0, 1, 0, 0, self.__time_step, 0, 0],
        #                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, self.__time_step, 0],
        #                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, self.__time_step],
        #                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # Матрица-связь вектора состояния с показаниями датчиков
        self.__filter.H = np.eye(len(self.__x))
        # Ковариационная матрица ошибки модели
        # self.__filter.Q = Q_discrete_white_noise(dim=len(self.__x), dt=self.__time_step, var=self.__process_noise)
        # Ковариационная матрица ошибки измерений
        self.__filter.R = np.diag(self.__r_x)
        # Ковариационная матрица состояния
        self.__filter.P = np.diag(self.__p_x)
        self.__filtered_state.append(self.__x)

    def __compute_z(self, deltas: np.array):
        return np.array([self.__filtered_state[-1][0] + deltas[0],
                         self.__filtered_state[-1][1] + deltas[1],
                         self.__filtered_state[-1][2] + deltas[2],
                         deltas[0] / deltas[-1],
                         deltas[1] / deltas[-1],
                         deltas[2] / deltas[-1],
                         self.__filtered_state[-1][6] + deltas[3],
                         self.__filtered_state[-1][7] + deltas[4],
                         self.__filtered_state[-1][8] + deltas[5],
                         deltas[3] / deltas[-1],
                         deltas[4] / deltas[-1],
                         deltas[5] / deltas[-1]]), deltas[-1]

    def __call__(self, deltas: np.array) -> Tuple[np.array, np.array]:
        """
        Производит очередную итерацию оценки вектора состояния по переданному вектору наблюдений.
        :param deltas: Сырой вектор наблюдений.
        :return: Кортеж вида (Последняя оценка вектора состояния, ковариационная матрица ошибок этой оценки).
        """

        z, time_step = self.__compute_z(deltas=deltas)
        if len(z) != self.__dim_z:
            raise VectorSizeMismatch(f"Передан вектор длиной {len(z)}, "
                                     f"ожидается {self.__dim_z}")

        self.__filter.F = np.array([[1, 0, 0, time_step, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, time_step, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, time_step, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, time_step, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, time_step, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, time_step],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.__filter.Q = Q_discrete_white_noise(dim=2,
                                                 dt=time_step,
                                                 var=self.__process_noise,
                                                 block_size=6,
                                                 order_by_dim=False)

        self.__filter.predict()
        self.__filter.update(z)
        self.__filtered_state.append(self.__filter.x)
        self.__covariance_history.append(self.__filter.P)
        self.__z_history.append(self.__filter.z)
        return self.__filtered_state[-1], self.__covariance_history[-1]

    def show(self):
        for i in range(len(self.__x)):
            plt.title("Kalman filter")
            plt.plot([x[i] for x in self.__z_history], label=f"Измерение {self.__descriptions[i]}", color="#99AAFF")
            plt.plot([x[i] for x in self.__filtered_state], label=f"Оценка фильтра {self.__descriptions[i]}", color="#224411")
            plt.legend()
            plt.show()
