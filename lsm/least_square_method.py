import numpy as np
from numpy.linalg import inv

def lsm(A: np.ndarray, B: np.array):
    return np.dot(np.dot(inv(np.dot(A.transpose(),A)),A.transpose()),B)

def get_error(A: np.ndarray, B: np.array, X: np.array, calculate_error):
    calculated = np.dot(A, X)
    return calculate_error(B, calculated)

def seq_lsm(pts1, pts2, rotation_matrix, dist_threshold = 35):
    emissions = []
    emissions_count = 1
    A = []
    B = []

    emissions = [True]*len(pts1)

    errors = [[], [], []]
    deltas = []
    emissions_count = emissions.count(True)

    # Calculating matrixes for LSM
    A, B = seq_lsm_init_matrixes(pts1, pts2, rotation_matrix, emissions)
    X = lsm(A, B)

    deltas = np.dot(A, X) - B
    errors = [[], [], []]
    elements_amount = deltas.shape[0]//3

    i = 0
    #Errors calculation
    for element_index in range(len(emissions)):
        if emissions[element_index]:
            flag = True
            for k in range(3):
                if (abs(deltas[i*3+k]) > dist_threshold):
                    flag = False
                    emissions_count += 1
                    break
                errors[k].append(deltas[i*3+k])
            emissions[element_index] = flag
            i += 1
        else:
            emissions[element_index] = False

    elements_amount = emissions.count(True)
    if elements_amount < 10:
        return [], [], [], []

    # Calculating 
    error_EV = [sum(error) / elements_amount for error in errors]
    error_DP = [sum([((delta - ev)**2)/(elements_amount-1) for delta in error]) for ev, error in zip(error_EV,errors)]

    return X, emissions, error_EV, error_DP

def seq_lsm_init_matrixes(pts1, pts2, rotation_matrix, emissions):
    A = []
    B = []
    for pt1, pt2 in zip(pts1[emissions], pts2[emissions]):
        tmp1 = np.ones((3,1)); tmp1[0] = pt1[0]; tmp1[1] = pt1[1]
        tmp1 = np.dot(rotation_matrix, tmp1)
        tmp2 = np.ones((3,1)); tmp2[0] = pt2[0]; tmp2[1] = pt2[1]

        for i in range(3):
            B.append(tmp2[i] - tmp1[i])

        if len(A) != 0:
            A = np.append(A, np.eye(3,3), axis=0)
        else:
            A = np.eye(3)
    B = np.array(B)
    return A, B

def map_lsm(pts1, pts2, dist_threshold = 35):
    emissions = []
    emissions_count = 1
    A = []
    B = []

    emissions = [True]*len(pts1)

    errors = [[], [], []]
    deltas = []
    emissions_count = emissions.count(True)

    # While there are points that are not emissions
    while emissions.count(True) > 0:
        # Calculating matrixes for LSM
        A, B = map_lsm_init_matrixes(pts1, pts2, emissions)
        X = lsm(A, B)

        deltas = np.dot(A, X) - B
        errors = [[], []]
        elements_amount = deltas.shape[0]//2

        i = 0
        #Errors calculation
        for element_index in range(len(emissions)):
            if emissions[element_index]:
                flag = True
                for k in range(2):
                    if (abs(deltas[i*2+k]) > dist_threshold):
                        flag = False
                        emissions_count += 1
                        break
                    errors[k].append(deltas[i*2+k])
                emissions[element_index] = flag
                i += 1
            else:
                emissions[element_index] = False

        # If there are emissions -> recalculate vector
        if emissions_count != 0:
            emissions_count = 0
            continue

        elements_amount = emissions.count(True)

        if elements_amount < 10:
            return [], [], [], []

        # Calculating
        error_EV = [sum(error) / elements_amount for error in errors]
        error_DP = [sum([((delta - ev)**2)/(elements_amount-1) for delta in error]) for ev, error in zip(error_EV,errors)]

        return X, emissions, error_EV, error_DP

    return [], [], [], []

def map_lsm_init_matrixes(pts1, pts2, emissions):
    A = []
    B = []
    for pt1, pt2 in zip(pts1[emissions], pts2[emissions]):
        A.append([pt1[0], -pt1[1], 1, 0])
        A.append([pt1[1],  pt1[0], 0, 1])
        B.append(pt2[0])
        B.append(pt2[1])

    A = np.array(A)
    B = np.array(B)

    return A, B