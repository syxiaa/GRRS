import pandas as pd
import numpy as np
import random
import sys
from sklearn.preprocessing import MinMaxScaler
sys.setrecursionlimit(100000)


def GRRS():
    red = []  # red is the result of reduction and initialized to an empty set
    n = U.shape[0]   # Number of samples
    Threshold = t / 2  # Voting threshold
    temp_pos_count = 0  # The number of positive regions each iteration

    # Iterate until no attributes can be added
    iteration = 1
    while iteration:
        rem = list(set(C)-set(red))   # Remaining attribute set
        temp_feature = -1
        for feature in rem:
            red.append(feature)
            positive_domains_list = np.zeros(n)
            for i in range(t):  # Carry out t times space divisions
                space_division(U, red, positive_domains_list, 0)
            pos_count = sum(x > Threshold for x in positive_domains_list)   # vote
            if pos_count > temp_pos_count:
                temp_feature = feature
                temp_pos_count = pos_count
            red.remove(feature)
        if temp_feature != -1:  # If the number of positive domains increases
            red.append(temp_feature)
        else:  # Otherwise, the iteration ends
            return red


def space_division(U, red, positive_domains_list, r_real):
    # If all samples have the same condition attributes, stop dividing
    same_flag = 1
    for attr in red:
        if len(set(U[:, attr])) != 1:
            same_flag = 0
            break
    if same_flag:
        if r_real >= r:  # If the neighborhood  radius is greater than r, then all samples in the space are set to be positive regions
            for point in list(U[:, -1]):
                positive_domains_list[int(point)] += 1
        return

    # If the labels of the samples are the same, add 1 to the neighborhood radius
    if r_real > 0:
        r_real += 1
    else:
        if len(set(U[:, -2])) == 1:
            r_real += 1

    # Randomly select an attribute to divide the current space
    temp_red = list(red)
    len_attributeset = 1
    while len_attributeset == 1:
        index = random.choice(temp_red)
        len_attributeset = len(set(U[:, int(index)]))
        temp_red.remove(index)
    index = int(index)

    # The dividing point is the median of all values under the attribute
    attributeset = list(set(U[:, index]))
    attributeset.sort()
    value = attributeset[int(len(attributeset)/2)]

    # Continue to divide the space
    ldata = U[np.where(U[:, index] < value)]
    space_division(ldata, red, positive_domains_list, r_real)
    rdata = U[np.where(U[:, index] >= value)]
    space_division(rdata, red, positive_domains_list, r_real)


if __name__ == '__main__':
    file_path = r'D:/credit.csv'   # dataset, each row is a sample, and the label is placed in the first column
    df = pd.read_csv(file_path, header=None)
    data = df.values
    numberSample, numberAttribute = data.shape
    minMax = MinMaxScaler()
    U = np.hstack((minMax.fit_transform(data[:, 1:]), data[:, 0].reshape(numberSample, 1)))  # data
    C = list(np.arange(0, numberAttribute - 1))  # Conditional attributes
    D = list(set(U[:, -1]))  # Decision attributes
    index = np.array(range(0, numberSample)).reshape(numberSample, 1)  # Index column
    U = np.hstack((U, index))  # Add index column
    t = 5  # Number of space divisions
    for r in range(1, 10):  # Neighborhood radius
        print('Neighborhood radius: ', r)
        red = SDNRS()
        print('red：', red)