import numpy as np
import pandas as pd
import random
import warnings

warnings.filterwarnings("ignore")  # 忽略警告
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)


# 节点类
class Node(object):
    def __init__(self, type=None, label_parent=None, label=None, parent=None, r_real=None, points=None, lchild=None,
                 rchild=None):
        self.type = type  # Node type 0: leaf 1: Branch
        self.label_parent = label_parent  # Label of parent node
        self.label = label  # Node label
        self.parent = parent  # Keep the parent class samples of each level of each node
        self.r_real = r_real
        self.points = points
        self.lchild = lchild  # Left subtree
        self.rchild = rchild  # Right subtree


def GRRS(U, red, label_parent, r_real, Parent, temp_equivalence_class, positive_domains_list):
    red = list(red)  # red is the result of reduction and initialized to an empty set
    node = Node()
    node.label_parent = label_parent
    node.points = list(U[:, -1])
    node.parent = Parent
    if U.shape[0] == 0:
        return
    # If there is only one sample, the division is stopped and the node is marked as a leaf node
    if U.shape[0] == 1 and r_real >= r:
        lk = []
        for f_number in node.parent:
            lk.append(int(f_number))
        temp_equivalence_class.append(lk)
        return node
    if U.shape[0] == 1 and r_real == 0:
        node.type = 0
        lk = []
        for f_number in node.points:
            lk.append(int(f_number))
        temp_equivalence_class.append(lk)
        return node
    # If the condition attributes of all samples are consistent, the division is stopped and the node is marked as a leaf node
    same_flag = 1
    for attr in red:
        if len(set(U[:, attr])) != 1:
            same_flag = 0
            break
    if same_flag and r_real == 0:
        node.type = 0
        lk = []
        for f_number in node.points:
            lk.append(int(f_number))
        temp_equivalence_class.append(lk)
        return node
    # If the labels of all samples are consistent and the current radius meets the radius we require, the division will
    # be stopped, the node will be marked as a leaf node,
    # and the samples contained in the current equivalence class are positive domain samples
    if r_real > 0:
        r_real += 1
        # When the radius is greater than 1, the parent node equivalence class of the node is the sample of the node with r = 1
        node.parent = Parent
        # If the radius of the current node is larger than our set neighborhood radius, the division will be ended
        # and the parent positive domain equivalent class of the current node will be added to temp_ equivalence_ class
        if r_real >= r:
            for point in list(U[:, -1]):
                positive_domains_list[int(point)] += 1
            node.r_real = r
            node.type = 0
            len_temp = len(temp_equivalence_class)
            l = node.parent
            if len_temp > 0:
                if l not in temp_equivalence_class:
                    lk = []
                    for f_number in node.parent:  # Convert the sequence value of the equivalence class into an integer
                        lk.append(int(f_number))
                    temp_equivalence_class.append(lk)
            else:
                lk = []
                for f_number in node.parent:
                    lk.append(int(f_number))
                temp_equivalence_class.append(lk)
            return node
    else:
        if len(set(U[:, -2])) == 1:
            r_real += 1
            node.parent = node.points
            if r_real >= r:
                for point in list(U[:, -1]):
                    positive_domains_list[int(point)] += 1
                node.r_real = r
                node.type = 0
                len_temp = len(temp_equivalence_class)
                l = node.parent
                if len_temp > 0:
                    if l not in temp_equivalence_class:
                        lk = []
                        for f_number in node.parent:
                            lk.append(int(f_number))
                        temp_equivalence_class.append(lk)
                else:
                    lk = []
                    for f_number in node.parent:
                        lk.append(int(f_number))
                    temp_equivalence_class.append(lk)
                return node
    # If the appeal conditions are not met, it is set as a branch node
    node.type = 1
    # If the samples are all one type of labels, the label of the branch node is set to this label
    label = list(set(U[:, -2]))
    if len(label) == 1:
        node.label = U[0, -2]
    else:
        node.label = 99999

    # Select the optimal partition attribute
    index, value = find_random_attribute(U, red)
    # Filter out the rows whose value in the index column is less than or equal to value, and generate the left subtree
    ldata = U[np.where(U[:, index] < value)]
    node.lchild = GRRS(ldata, red, node.label, r_real, node.parent, temp_equivalence_class,
                       positive_domains_list)
    # Filter out the rows whose value in the index column is greater than value and generate a right subtree
    rdata = U[np.where(U[:, index] >= value)]
    node.rchild = GRRS(rdata, red, node.label, r_real, node.parent, temp_equivalence_class,
                       positive_domains_list)
    return node


def find_random_attribute(U, red):
    temp_red = list(red)
    len_attributeset = 1
    while len_attributeset == 1:  # If the currently divided data set has the same value on a certain attribute, change another attribute
        index = random.choice(temp_red)
        len_attributeset = len(set(U[:, int(index)]))
        temp_red.remove(index)
        if len(temp_red) == 0:
            break
    index = int(index)

    # The dividing point is the median of all values under the attribute
    attributeset = list(set(U[:, index]))

    attributeset.sort()
    # Take the attribute value of the current equivalent sample corresponding to the current random attribute, sort it,
    # and take the random attribute value corresponding to the median as the partition node
    value = attributeset[int(len(attributeset) / 2)]
    return index, value


def GRRSCTree(re, train_data_cl, test_data_cl):
    tree_count = 10
    numberSample, numberAttribute = test_data_cl.shape  # Sample number and label column of test data
    flag = np.zeros(numberSample)
    numberSample1, numberAttribute1 = train_data_cl.shape  # Sample number and label column of training data
    list_forest_equivalence_class = []
    positive_domains_list = np.zeros(numberSample1)  # Initialization is 0
    for k in range(tree_count):
        temp_equivalence_class = []
        GRRS(train_data_cl, re, None, 0, train_data_cl, temp_equivalence_class, positive_domains_list)
        list00 = []
        for j in temp_equivalence_class:
            for k in j:
                list00.append(k)
        count_sum = 0
        for i in temp_equivalence_class:
            count_sum += len(i)
        list_forest_equivalence_class.append(temp_equivalence_class)
        acc = 0
    positive_domains_list = [1 if i >= 5 else 0 for i in positive_domains_list]  # Statistical positive domain samples
    forest_intersec_equivalence_class = []  # Equivalence class used to preserve forest intersection
    # Obtain the intersection of positive field equivalence classes in ten trees corresponding to each positive field sample
    for index, value in enumerate(positive_domains_list):
        single_forest_equivalence_class = []  # Save the positive domain equivalent class of the ten trees where the current positive domain node is located
        if value == 1:  # If the current node is a positive domain node
            for tree_equivalence_class in list_forest_equivalence_class:
                for single_tree_equivalence_class in tree_equivalence_class:
                    if index in single_tree_equivalence_class:
                        if len(set(
                                [train_data_cl[single_index, -2] for single_index in
                                 single_tree_equivalence_class])) == 1:
                            single_forest_equivalence_class.append(single_tree_equivalence_class)
                            break
            # Find the intersection of positive domain equivalence classes of current multiple trees
            length = len(single_forest_equivalence_class)
            a = single_forest_equivalence_class[0]
            for m in range(1, length):
                a = list(set(a).intersection(set(single_forest_equivalence_class[m])))
            forest_intersec_equivalence_class.append(a)
            for a_index in a:
                positive_domains_list[a_index] = 0
    # Using the forest equivalence class to find the boundary region, that is,
    # the difference between the forest equivalence class and the training sample
    list_train_samples = []  # Index set of training samples
    for i in range(numberSample1):
        list_train_samples.append(i)
    list_intersc_equclass_samples = []  # Samples of equivalence classes of intersection positive fields
    for i in forest_intersec_equivalence_class:
        for j in i:
            list_intersc_equclass_samples.append(j)
    return forest_intersec_equivalence_class


from sklearn.model_selection import KFold

keys = ['Algerian_forest_fires_dataset_UPDATE']
for d in range(len(keys)):
    df = pd.read_csv("E:\\dataset\\" + keys[d] + ".csv", header=None)
    # df = df.reindex(np.random.permutation(df.index))
    re = []
    data = df.values
    numberSample, numberAttribute = data.shape
    data = np.hstack((data[:, 1:], data[:, 0].reshape(numberSample, 1)))
    # Create a list to hold the number sequence from 1 to numbersample
    orderAttribute = np.array([i for i in range(0, numberSample)]).reshape(numberSample, 1)
    data = np.hstack((data, orderAttribute))
    kf = KFold(n_splits=5, shuffle=False, random_state=0)
    acc = 0
    line = [4,7]
    r = 1
    split_data = kf.split(data)  # Cross segmentation data
    for split_item in split_data:
        train_data = data[split_item[0]]
        test_data = data[split_item[1]]
        numberS, numberA = train_data.shape
        orderAttribute = np.array([i for i in range(0, numberS)]).reshape(numberS, 1)
        train_data = np.hstack((train_data[:, :numberA - 1], orderAttribute))
        re1 = []
        for i in line:
            re1.append(i)
            # Generate the Granular-rectangle equivalence class division result of each layer
            GRRSCTree(re1, train_data, test_data)
        break
