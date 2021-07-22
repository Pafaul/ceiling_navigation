import math

import numpy as np
from cv2 import cv2


class DefaultOpticalFlow:
    def __init__(self):
        raise NotImplementedError()

    def get_optical_flow(self, first_image: np.ndarray, second_image: np.ndarray, block_size: list = [8, 8],
                         search_window_size: list = [1, 1]):
        """
        Get optical flow from the image
        :param first_image: First image
        :param second_image: Second image
        :param block_size: Size of block to use
        :param search_window_size: Size of search window in blocks
        :return:
        """
        raise NotImplementedError()

    def draw_optical_flow(self, image: np.ndarray, result_dict: dict):
        raise NotImplementedError()

    def filter_optical_flow(self, result_dict: dict):
        raise NotImplementedError()


class CorrelationOpticalFlow(DefaultOpticalFlow):
    def __init__(self):
        pass

    def get_optical_flow(self, first_image: np.ndarray, second_image: np.ndarray, block_size: list = [8, 8],
                         search_window_size: list = [1, 1]) -> dict:
        """
        Get optical flow (offsets of image blocks in pixels)
        :param first_image: first image (that was taken earlier than second)
        :param second_image: second image
        :param block_size: size of blocks to use
        :param search_window_size: size of search window in blocks
        :return: dictionary with elements: number_of_blocks, search_window_size, offsets
        """
        block_size_x, block_size_y = block_size
        search_window_size_x = math.floor((block_size_x * search_window_size[0])/2)
        search_window_size_y = math.floor((block_size_y * search_window_size[1])/2)
        x_blocks = math.floor((first_image.shape[1] - search_window_size_x*2)/block_size_x)
        y_blocks = math.floor((first_image.shape[0] - search_window_size_y*2)/block_size_y)
        offsets = []
        for block_number_y in range(0, y_blocks):
            for block_number_x in range(0, x_blocks):
                offset = self.__find_maximum_cv2(
                    second_image[
                        block_number_y*block_size_y:
                        (block_number_y+1)*block_size_y + search_window_size_y*2,
                        block_number_x*block_size_x:
                        (block_number_x+1)*block_size_x + search_window_size_x*2
                    ],
                    first_image[
                        search_window_size_y + block_number_y*block_size_y:
                        search_window_size_y + (block_number_y+1)*block_size_y,
                        search_window_size_x + block_number_x*block_size_x:
                        search_window_size_x + (block_number_x+1)*block_size_x
                    ],
                    block_size
                )
                offsets.append(offset)
        return_value = dict()
        return_value['number_of_blocks'] = [x_blocks, y_blocks]
        return_value['block_size'] = block_size
        return_value['search_window_size'] = [search_window_size_x, search_window_size_y]
        return_value['offsets'] = offsets
        return return_value

    def draw_optical_flow(self, image: np.ndarray, result_dict: dict):
        number_of_blocks = result_dict['number_of_blocks']
        search_window_size = result_dict['search_window_size']
        block_size = result_dict['block_size']
        offsets = result_dict['offsets']
        for x_block_number in range(0, number_of_blocks[0]):
            for y_block_number in range(0, number_of_blocks[1]):
                x_block_center = math.floor((x_block_number + 0.5)*block_size[0] + search_window_size[0])
                y_block_center = math.floor((y_block_number + 0.5)*block_size[1] + search_window_size[1])
                delta = list(offsets[x_block_number + y_block_number * number_of_blocks[0]])
                if delta[0] != -1:
                    delta[0] -= search_window_size[0]
                    delta[1] -= search_window_size[1]
                else:
                    delta[0] = 0
                    delta[1] = 0

                cv2.line(
                    image,
                    (x_block_center, y_block_center),
                    (x_block_center + delta[0], y_block_center + delta[1]),
                    (0,),
                    thickness=2,
                    lineType=cv2.LINE_8
                )

        return image

    def filter_optical_flow(self, result_dict: dict):
        number_of_blocks = result_dict['number_of_blocks']
        offsets = result_dict['offsets']
        for x_block_number in range(1, number_of_blocks[0]-1):
            for y_block_number in range(1, number_of_blocks[1]-1):
                delta = list(offsets[x_block_number + y_block_number * number_of_blocks[0]])
                points = []
                points.append(list(offsets[x_block_number + (y_block_number - 1) * number_of_blocks[0]]))
                points.append(list(offsets[x_block_number + (y_block_number + 1) * number_of_blocks[0]]))
                points.append(list(offsets[x_block_number - 1 + y_block_number * number_of_blocks[0]]))
                points.append(list(offsets[x_block_number + 1 + y_block_number * number_of_blocks[0]]))
                vector_part = ((delta[0]**2 + delta[1]**2)**0.5)/3
                x_delta = sum([abs(p[0] - delta[0]) for p in points])
                y_delta = sum([abs(p[1] - delta[1]) for p in points])
                if x_delta >= vector_part and y_delta >= vector_part:
                    result_dict['offsets'][x_block_number + y_block_number * number_of_blocks[0]] = (-1, -1)
        pass

    def __find_maximum_cv2(self, image_part: np.ndarray, template: np.ndarray, template_size: list) -> list:
        res = cv2.matchTemplate(image_part, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_loc if max_val >= 0.45 else [-1, -1]

    def __find_maximum(self, image_part: np.ndarray, template: np.ndarray, template_size: list) -> list:
        """
        Find point on image where correlation value is maximum
        :param image_part: image part where to look for template
        :param template: image template
        :param template_size: template size
        :return: point with maximum correlation coordinates -> [x_max, y_max]
        """

        # inverted because of opencv indexes
        maximum_x_offset = image_part.shape[1] - template_size[0]
        maximum_y_offset = image_part.shape[0] - template_size[1]

        maximum_correlation_value = -2
        maximum_correlation_point = [-1, -1]
        for x_offset in range(0, maximum_x_offset):
            for y_offset in range(0, maximum_y_offset):
                correlation_value = self.__get_correlation_value(
                    image_part[y_offset: y_offset + template_size[1], x_offset: x_offset + template_size[0]],
                    template
                )
                # first_image = image_part[y_offset: y_offset + template_size[1], x_offset: x_offset + template_size[0]]
                # first_mean = first_image.mean()
                # template_mean = template.mean()
                # correlation_value = ((first_image - first_mean) * (template - template_mean)).mean()

                if correlation_value > maximum_correlation_value:
                    maximum_correlation_value = correlation_value
                    maximum_correlation_point = [x_offset, y_offset]

        return maximum_correlation_point

    def __get_correlation_value(self, first_image_part: np.ndarray, second_image_part: np.ndarray) -> float:
        """
        Get correlation value of two images
        :param first_image_part: first image
        :param second_image_part: second image
        :return: Correlation value
        """
        first_mean = first_image_part.mean()
        second_mean = second_image_part.mean()

        return ((first_image_part - first_mean) * (second_image_part - second_mean)).mean()


class DiffOpticalFlow(DefaultOpticalFlow):
    def __init__(self):
        pass

    def get_optical_flow(self, first_image: np.ndarray, second_image: np.ndarray, block_size: list = [8, 8],
                         search_window_size: list = [1, 1]):
        """
                Get optical flow (offsets of image blocks in pixels)
                :param first_image: first image (that was taken earlier than second)
                :param second_image: second image
                :param block_size: size of blocks to use
                :param search_window_size: size of search window in blocks
                :return: dictionary with elements: number_of_blocks, search_window_size, offsets
                """
        block_size_x, block_size_y = block_size
        search_window_size_x = math.floor((block_size_x * search_window_size[0]) / 2)
        search_window_size_y = math.floor((block_size_y * search_window_size[1]) / 2)
        x_blocks = math.floor((first_image.shape[1] - search_window_size_x * 2) / block_size_x)
        y_blocks = math.floor((first_image.shape[0] - search_window_size_y * 2) / block_size_y)
        offsets = []
        for block_number_y in range(0, y_blocks, 2):
            for block_number_x in range(0, x_blocks, 2):
                offset = self.__find_minimum(
                    second_image[
                        block_number_y * block_size_y:
                        (block_number_y + 1) * block_size_y + search_window_size_y * 2,
                        block_number_x * block_size_x:
                        (block_number_x + 1) * block_size_x + search_window_size_x * 2
                    ],
                    first_image[
                        search_window_size_y + block_number_y * block_size_y:
                        search_window_size_y + (block_number_y + 1) * block_size_y,
                        search_window_size_x + block_number_x * block_size_x:
                        search_window_size_x + (block_number_x + 1) * block_size_x
                    ],
                    block_size
                )
                offsets.append(offset)
        return_value = dict()
        return_value['number_of_blocks'] = [x_blocks, y_blocks]
        return_value['search_window_size'] = [search_window_size_x, search_window_size_y]
        return_value['offsets'] = offsets
        return return_value
        pass

    def draw_optical_flow(self, image: np.ndarray, result_dict: dict):
        pass

    def __find_minimum(self, image_part: np.ndarray, template: np.ndarray, template_size: list) -> list:
        """
        Find point on image where correlation value is maximum
        :param image_part: image part where to look for template
        :param template: image template
        :param template_size: template size
        :return: point with maximum correlation coordinates -> [x_max, y_max]
        """

        # inverted because of opencv indexes
        maximum_x_offset = image_part.shape[1] - template_size[0]
        maximum_y_offset = image_part.shape[0] - template_size[1]

        minimal_diff_value = 255*image_part.shape[1]*image_part.shape[0]
        minimum_point = [-1, -1]
        for x_offset in range(0, maximum_x_offset):
            for y_offset in range(0, maximum_y_offset):
                correlation_value = self.__calculate_diff(
                    image_part[y_offset: y_offset + template_size[1], x_offset: x_offset + template_size[0]],
                    template
                )

                if correlation_value < minimal_diff_value:
                    minimal_diff_value = correlation_value
                    minimum_point = [x_offset, y_offset]

        return minimum_point

    def __calculate_diff(self, first_image_part, second_image_part) -> float:
        """
        Get diff value of image blocks
        :param first_image_part: first image block
        :param second_image_part: second image block
        :return: diff measure
        """

        return sum(sum(abs(first_image_part - second_image_part)))
