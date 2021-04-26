import math
import operator
import copy
import random

# tree stuff
def compute_random_subset(values, num_values):
    shuffled = values[:] # shallow copy 
    random.shuffle(shuffled)
    return sorted(shuffled[:num_values])

def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
    return sample
# end tree stuff

# pa6 functions
def print_tree(header, tree, class_name, built_string):
    if tree[0] == 'Attribute':
        built_string += ' AND ' + tree[1] + ' == '
        print('AND', tree[1], '==', end=' ')
        for i in range(2, len(tree)):
            print_tree(header, tree[i], class_name, built_string)
            if i < len(tree) - 1:
                print(built_string, end='')
    elif tree[0] == 'Value':
        built_string += str(tree[1])
        print(tree[1], end=' ')
        print_tree(header, tree[2], class_name, built_string)
    else:
        print('THEN', class_name, '=', tree[1])

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == 'Attribute':
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2,len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match! recurse
                return tdidt_predict(header, value_list[2], instance)
    else: #leaf
        return tree[1] # leaf class label

def select_attribute(instances, available_attributes, attribute_domains, header):
    # for now we are going to select an attribute randomly
    # TODO: come back after you can build a tree with random
    #       attribute selection and replace with entropy
    if len(available_attributes) == 0:
        return []
    rand_index = random.randrange(0, len(available_attributes))
    class_labels = unique_index(instances, -1)

    entropies = []
    for attr in available_attributes:
        domain = attribute_domains[attr]
        # partition on each attribute (so 5,4,5 on first one)
        # 5 senior, 4 mid, 5 junior
        partitions = [[] for _ in domain]
        for row in instances:
            for i,val in enumerate(domain):
                if row[header.index(attr)] == val:
                    partitions[i].append(row)
        
        # calculate different class label distribution
        entropy_local = 0
        for partition in partitions:
            # 5 seniors
            entropy = 0
            counts = [0 for _ in class_labels]
            for row in partition:
                for i,label in enumerate(class_labels):
                    if row[-1] == label:
                        counts[i]+=1
            for count in counts:
                if count != 0:
                    entropy -= (count/sum(counts))*math.log(count/sum(counts), 2)
            entropy_local += len(partition)/len(instances)*entropy
        entropies.append(entropy_local)

    return available_attributes[entropies.index(min(entropies))]

def partition_instances(instances, split_attribute, attribute_domains, header):
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is level
    attribute_domain = attribute_domains[split_attribute] # ['Senior','Mid', 'Junior']
    attribute_index = header.index(split_attribute) # 0
    # Lets build a dictionary
    partitions = {} # key (attribute value): value (list of instances with this attribute value)
    for val in attribute_domain:
        partitions[val] = []
    
    for inst in instances:
        partitions[inst[attribute_index]].append(inst)
    return partitions

def tdidt(current_instances, available_attributes, attribute_domains, header):
    # basic approach (uses recursion!!):

    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes, attribute_domains, header)
    #print('splitting on', split_attribute)
    available_attributes.remove(split_attribute) # cannot split on same attr twice in a branch
    # python is pass by object reference!!
    tree = ['Attribute', split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    #print('partitions:', partitions)

    prev = []
    prev_instances = 0
    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        #print('working with partition for', attribute_value)
        value_subtree = ['Value', attribute_value]
        subtree = []
        # TODO: appending leaf nodes and subtrees appropriately to value_subtree
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition): # all same class checks if all the other values equal the first one
            subtree = ['Leaf', partition[0][-1], len(partition), len(current_instances)]
            value_subtree.append(subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            subtree = ['Leaf', majority_vote(partition), len(partition), len(current_instances)]
            value_subtree.append(subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            return ['Leaf', majority_vote(prev), len(prev), prev_instances]
        else:
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains.copy(), header.copy())
            value_subtree.append(subtree)
            # need to append subtree to value_subtree and appropriately append value_subtree to tree
        tree.append(value_subtree)
        prev = partition
        prev_instances = len(current_instances)

    return tree

def majority_vote(vals):
    if len(vals) == 0:
        return 0
    unique = []
    counts = []
    for row in vals:
        if row[-1] not in unique:
            unique.append(row[-1])
            counts.append(1)
        else:
            counts[unique.index(row[-1])] += 1
    return unique[counts.index(max(counts))]

def all_same_class(vals):
    c = vals[0][-1]
    for row in vals:
        if row[-1] != c:
            return False
    return True

def unique_index(vals, i):
    unique = []
    for row in vals:
        if row[i] not in unique:
            unique.append(row[i])
    return unique

# end pa6 functions

def mean(vals):
    return sum(vals)/len(vals)

def compute_slope_intercept(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return float(m), float(b)

def compute_euclidean_distance(v1, v2):
    if len(v1) > len(v2):
        return math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v2))]))
    else:
        return math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def scale(vals, test_vals):
    res = []
    max_vals = []
    min_vals = []
    for i in range(len(vals[0])):
        max_vals.append(max(get_column(vals, i)))
        min_vals.append(min(get_column(vals, i)))
    for row in vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-min_vals[i])/(max_vals[i]-min_vals[i]))
        res.append(curr)
    for row in test_vals:
        curr = []
        for i in range(len(row)):
            curr.append((row[i]-min_vals[i])/(max_vals[i]-min_vals[i]))
        res.append(curr)
    return res[:len(vals)], res[len(vals):]

def get_column(vals, i):
    return [val[i] for val in vals]

def kneighbors_helper(scaled_train, scaled_test, n_neighbors):
    # deep copy so you don't modify the original
    scaled_train = copy.deepcopy(scaled_train)
    scaled_test = copy.deepcopy(scaled_test)
    for i, instance in enumerate(scaled_train):
        # append the original row index
        instance.append(i)
        # append the distance
        dist = compute_euclidean_distance(instance[:-1], scaled_test)
        instance.append(dist)
    
    train_sorted = sorted(scaled_train, key=operator.itemgetter(-1))

    top_k = train_sorted[:n_neighbors]
    distances = []
    indices = []
    for row in top_k:
        distances.append(row[-1])
        indices.append(row[-2])
    
    return distances, indices

def get_label(labels):
    unique_labels = []
    for val in labels:
        if val not in unique_labels:
            unique_labels.append(val)
    counts = [0 for _ in unique_labels]
    for i, val in enumerate(unique_labels):
        for lab in labels:
            if val == lab:
                counts[i] += 1
    max_count = 0
    res = ''
    for i, val in enumerate(counts):
        if val > max_count:
            res = unique_labels[i]

    return res

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap the element at i with
        rand_index = random.randrange(0, len(alist)) # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def get_unique(vals):
    unique = []
    for val in vals:
        if val not in unique:
            unique.append(val)
    return unique

def group_by(x_train, y_train):
    unique = get_unique(y_train)
    grouped = [[] for _ in unique]
    for i, val in enumerate(y_train):
        for j, label in enumerate(unique):
            if val == label:
                grouped[j].append(i)
    return grouped

def get_from_folds(X_vals, y_vals, train_folds, test_folds):
    X_train = []
    y_train = []
    for row in train_folds:
        for i in row:
            X_train.append(X_vals[i])
            y_train.append(y_vals[i])

    X_test = []
    y_test = []
    for row in test_folds:
        for i in row:
            X_test.append(X_vals[i])
            y_test.append(y_vals[i])

    return X_train, y_train, X_test, y_test

def count_vals(vals):
    unique = get_unique(vals)
    res = [0 for _ in unique]
    for v in vals:
        for i, u in enumerate(unique):
            if v == u:
                res[i] += 1
    return res

def get_column(table, i):
    res = []
    for row in table:
        res.append(row[i])
    return res

def posteriors(X_train, y_train, priors):
    res = {}
    cols = []
    attrs = []

    for i in range(len(X_train[0])):
        cols.append(get_column(X_train, i))
    for row in cols:
        attrs.append(get_unique(row))

    for val, _ in priors.items():
        res[val] = {}
        for i in range(len(attrs)):
            res[val][i] = {}

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            class_label = y_train[i]
            attr_val = X_train[i][j]
            if attr_val in res[class_label][j]:
                res[class_label][j][attr_val] = ((res[class_label][j][attr_val]*(priors[class_label]*len(y_train))) + 1)/(priors[class_label]*len(y_train))
            else:
                res[class_label][j][attr_val] = 1/(priors[class_label]*len(y_train))

    return res

def priors(y_train):
    unique = get_unique(y_train)
    res = {}
    for val in unique:
        res[val] = 0
    for val in y_train:
        for u in unique:
            if val == u:
                res[u] += 1
    for u in unique:
        res[u] /= len(y_train)
    return res

def get_prediction_index(vals):
    max_index = 0
    for i in range(len(vals)):
        if vals[i] > vals[max_index]:
            max_index = i
    return max_index
