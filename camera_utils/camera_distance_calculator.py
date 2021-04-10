from math import tan
from typing import Any

class CameraDistanceCalculator:
    def __init__(self, h: float, alphaX: float, alphaY: float, px: int, py: int):
        self.__h = h
        self.__alphaX = alphaX
        self.__alphaY = alphaY
        self.__px = px
        self.__py = py

    def calculate_dist(self, px: int, py: int) -> list:
        lx = self.__h*tan(self.__alphaX/2)*2
        ly = self.__h*tan(self.__alphaY/2)*2
        return [px/self.__px*lx, py/self.__py*ly]
