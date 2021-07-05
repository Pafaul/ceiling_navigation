from math import tan
from typing import Any


class CameraDistanceCalculator:
    def __init__(self, h: float, alphaX: float, alphaY: float, px: int, py: int):
        """
        Initialization of camera distance calculator (x, y in pixels to x, y in meters)
        :param h: height
        :param alphaX: POV along X axis
        :param alphaY: POV along Y axis
        :param px: resolution along X axis
        :param py: resolution along Y axis
        """
        self.__h = h
        self.__alphaX = alphaX
        self.__alphaY = alphaY
        self.__px = px
        self.__py = py

    def calculate_dist(self, px: int, py: int) -> list:
        """
        Calculate distance from pixel distance
        :param px: Distance in pixels along X axis
        :param py: Distance in pixels along Y axis
        :return:
        """
        lx = self.__h * tan(self.__alphaX / 2) * 2
        ly = self.__h * tan(self.__alphaY / 2) * 2
        return [px / self.__px * lx, py / self.__py * ly]
