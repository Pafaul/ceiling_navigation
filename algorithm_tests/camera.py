from math import tan, sqrt
import numpy as np


def scalar(v1, v2):
    return sum(v1*v2)


class Camera:
    def __init__(self, resolution: list, focal_length: float, aov: list, position: np.ndarray):
        """
        Camera object constructor
        :param resolution: image resolution
        :param focal_length: camera's focal length
        :param position: camera position
        """
        self.resolution = resolution
        self.focal_length = focal_length
        self.pixels_per_meter = [0, 0]
        self.position = position
        self.aov = aov
        self.update_parameters()

    def update_parameters(self):
        # pixels per meter
        self.pixels_per_meter[0] = self.resolution[0] / (2 * self.position[2] * tan(self.aov[0] / 2))
        self.pixels_per_meter[1] = self.resolution[1] / (2 * self.position[2] * tan(self.aov[1] / 2))

    def get_current_optical_axis(self, rotation_matrix: np.ndarray, project: bool = True) -> list:
        """
        Project camera center to plane
        1. Get plane equation
        2. Calculate current optical axis
        3. Calculate parameter t: t = - (A*x + B*y + C*z + D)/(A*m + B*p + C*l)
        4. Get current optical center
        :param rotation_matrix:
        :return:
        """
        # optical_axis = np.zeros([3, 1])
        # optical_axis[2] = -self.position[2]
        # optical_axis = rotation_matrix.dot(optical_axis)
        # optical_axis += self.position
        # optical_axis = rotation_matrix.dot(-self.position)
        # optical_axis += self.position
        optical_axis = self.position.copy()
        optical_axis[2] = 0
        v1 = np.zeros([3, 1])
        v1[2] = -self.position[2]
        v2 = rotation_matrix.dot(v1)
        delta = v2 - v1
        optical_axis += delta
        if project:
            optical_axis *= (self.position[2] + optical_axis[2])/(self.position[2])

        return optical_axis

    def get_camera_fov_in_real_coordinates(self, rotation_matrix: np.ndarray, project: bool = True, project_camera_axis: bool = True) -> list:
        points = [np.zeros([3, 1]) for _ in range(4)]
        meters = [
            self.resolution[0] / self.pixels_per_meter[0] / 2,
            self.resolution[1] / self.pixels_per_meter[1] / 2
        ]

        points[0][0] = -meters[0]
        points[0][1] = -meters[1]
        points[1][0] = -meters[0]
        points[1][1] =  meters[1]
        points[2][0] =  meters[0]
        points[2][1] =  meters[1]
        points[3][0] =  meters[0]
        points[3][1] = -meters[1]

        result_points = []

        current_image_center_position = self.get_current_optical_axis(rotation_matrix, project=project_camera_axis)

        for p in points:
            if project:
                tmp = p.copy()
                tmp = rotation_matrix.dot(tmp)
                tmp = tmp * (self.position[2] / (self.position[2] - tmp[2]))
                tmp = tmp + current_image_center_position
                result_points.append(tmp)
            else:
                result_points.append(rotation_matrix.dot(p) + current_image_center_position)

        return result_points

    def calculate_if_keypoints_visible(self, keypoints: list, rotation_matrix: np.ndarray) -> list:
        """
        Check if keypoints are visible by camera.
        :param keypoints: known keypoints
        :param rotation_matrix: current rotation matrix
        :return:
        """
        camera_coordinates = self.get_camera_fov_in_real_coordinates(rotation_matrix)
        O_hatch_A = camera_coordinates[3] - camera_coordinates[2]  # image axis Ox
        O_hatch_B = camera_coordinates[1] - camera_coordinates[2]  # image axis Oy

        mask = []
        for kp in keypoints:
            vector_M = kp - camera_coordinates[2]
            vector_M[2] = 0
            if (0 <= scalar(O_hatch_A, vector_M) <= scalar(O_hatch_A, O_hatch_A)) and \
                (0 <= scalar(O_hatch_B, vector_M) <= scalar(O_hatch_B, O_hatch_B)):
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def project_keypoint(self, keypoint: np.ndarray) -> np.ndarray:
        """
        Project keypoint to the surface
        :param keypoint:
        :return:
        """
        tmp = keypoint.copy()
        tmp = tmp * self.position[2] / tmp[2]
        return tmp

    def get_keypoint_image_position(self, keypoint: np.ndarray, camera_fov: list) -> np.ndarray:
        """
        Get keypoints position in image
        :param keypoint:
        :param camera_fov:
        :return:
        """
        tmp = np.ndarray([2, 1])
        tmp_res = np.ndarray([2, 1])
        tmp[0] = keypoint[0] - self.position[0]
        tmp[1] = keypoint[1] - self.position[1]
        # Calculate Ox and Oy in the image coordinates
        O_hatch_A = (camera_fov[3] - camera_fov[2])
        O_hatch_A /= np.linalg.norm(O_hatch_A) # image axis Ox
        O_hatch_B = (camera_fov[1] - camera_fov[2])
        O_hatch_B /= np.linalg.norm(O_hatch_B) # image axis Oy
        # just add half of image resolution, i thought that vector substraction will be enough
        tmp_res[0] = int(scalar(tmp[0:2], O_hatch_A[0:2]) * self.pixels_per_meter[0] + self.resolution[0] / 2)
        tmp_res[1] = int(scalar(tmp[0:2], O_hatch_B[0:2]) * self.pixels_per_meter[1] + self.resolution[1] / 2)
        return tmp_res

    def get_keypoint_image_position_from_plane(self, keypoint: np.ndarray, camera_fov: list) -> np.ndarray:
        """
        Get kypoint position on image from plane coordinates
        :param keypoint:
        :param camera_fov:
        :return:
        """
        vector_CD = camera_fov[3] - camera_fov[2] # along x axis
        vector_CB = camera_fov[1] - camera_fov[2] # along y axis
        vector_CD /= np.linalg.norm(vector_CD)    # normalizing vectors so we can project vectors
        vector_CB /= np.linalg.norm(vector_CB)
        vector_CM = keypoint - camera_fov[2]      # from image (0, 0) to keypoint
        tmp_res = np.ndarray([2, 1])

        tmp_res[0] = int(scalar(vector_CM, vector_CD) * self.pixels_per_meter[0])
        tmp_res[1] = int(scalar(vector_CM, vector_CB) * self.pixels_per_meter[1])
        return tmp_res

    def calculate_keypoint_camera_position(self, keypoints: list, rotation_matrix: np.ndarray) -> list:
        """
        Calculate keypoints position on image
        :param keypoints:
        :param rotation_matrix:
        :return:
        """
        keypoints_camera_positions = []
        fov = self.get_camera_fov_in_real_coordinates(rotation_matrix)
        mask = self.calculate_if_keypoints_visible(keypoints, rotation_matrix)
        for kp, is_visible in zip(keypoints, mask):
            if is_visible:
                tmp = kp.copy()
                tmp[2] = self.position[2] - tmp[2]
                tmp = self.project_keypoint(tmp)
                tmp = self.get_keypoint_image_position(tmp, fov)
                keypoints_camera_positions.append(tmp)
            else:
                keypoints_camera_positions.append(np.zeros([2, 1]))

        return [keypoints_camera_positions, mask]

    def get_straight_line_equation(self, point1: np.ndarray, point2: np.ndarray) -> list:
        """
        Calculate straight line equation in format:
        (x - x0)/m = (y - y0)/n = (z - z0)/p
        Uses two points for this
        (m, n, p) - vector from point1 to point2
        :param point1:
        :param point2:
        :return: [point of line, vector]
        """
        vector = point2 - point1
        return [point1, vector]

    def get_plane_equation(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> list:
        """
        Calculate plane equation in format:
        A*x + B*y + C*z + D = 0
        Steps:
        1. Calculate two vectors A and B
        2. Multiply two vectors A and B -> get perpendicular vector C
        3. Normalize perpendicular vector C
        4. Calculate parameter D from equation: A(x - x0) + B(y - y0) + C(z - z0) = 0
        :param point1:
        :param point2:
        :param point3:
        :return:
        """
        vector_A = (point2 - point1).reshape([3])
        vector_B = (point3 - point1).reshape([3])
        vector_C = np.cross(vector_A, vector_B)
        D = -(vector_C[0] * point1[0] + vector_C[1] * point1[1] + vector_C[2] * point1[2])
        return [vector_C[0], vector_C[1], vector_C[2], D]

    def project_keypoint_to_plane(self, point: np.ndarray, plane_parameters: list) -> list:
        """
        Project keypoint to plane
        1. Get perpendicular straight line equation: (x - x0)/m = (y - y0)/n = (z - z0)/p
        2. Calculate parameter t: t = - (A*x + B*y + C*z + D)/(A*m + B*p + C*l)
        3. Get line and plane intersection -> get point on plane
        :param point:
        :param plane_parameters:
        :return:
        """
        A, B, C, D = plane_parameters
        line_point, vector = self.get_straight_line_equation(point, self.position)
        t = - (A*line_point[0] + B*line_point[1] + C*line_point[2] + D)/(A*vector[0] + B*vector[1] + C*vector[2])
        result_point = np.ndarray([3, 1])
        result_point[0] = point[0] + t*vector[0]
        result_point[1] = point[1] + t*vector[1]
        result_point[2] = point[2] + t*vector[2]
        return result_point

    def distance_from_point_to_plane(self, point: np.ndarray, plane_parameters: list) -> list:
        """
        Get distance from point to plane using following equation:
        d = |A*x + B*y + C*z + D|/sqrt(A^2 + B^2 + C^2)
        Distance is used to project keypoints to plane
        :param point:
        :param plane_parameters:
        :return:
        """
        return abs(
            plane_parameters[0]*point[0] + plane_parameters[1] * point[1] + plane_parameters[2]*point[2] + plane_parameters[3]
        )/sqrt(plane_parameters[0]**2 + plane_parameters[1]**2 + plane_parameters[2]**2)

    def calculate_keypoint_projections_to_plane(self, keypoints: list, rotation_matrix: np.ndarray):
        """
        Calculate keypoints position on image using plane projections
        1. Calculate camera's fov without projection of angle points +
        2. Get plane equation                                        +
        3. Get straight line equation from keypoint to camera
        3. Calculate keypoint's position on plane                    +
        4. Convert keypoint's position to position on image          +
        5. Check whether point is on image or not                    +
        6. Return result projections and mask                        +
        :param keypoints: list of keypoints
        :param rotation_matrix: rotation matrix used to rotate camera, angles must be less than 90 degrees
        :return:
        """

        camera_fov = self.get_camera_fov_in_real_coordinates(rotation_matrix, project=False, project_camera_axis=False)
        plane_parameters = self.get_plane_equation(camera_fov[1], camera_fov[0], camera_fov[2])
        mask = []
        result_keypoints = []
        for kp in keypoints:
            tmp = self.project_keypoint_to_plane(kp, plane_parameters)
            tmp = self.get_keypoint_image_position_from_plane(tmp, camera_fov)
            if (0 <= tmp[0] < self.resolution[0]) and (0 <= tmp[1] < self.resolution[1]):
                result_keypoints.append(tmp)
                mask.append(True)
            else:
                result_keypoints.append(np.ones([2, 1]) * -1)
                mask.append(False)

        return [result_keypoints, mask]



