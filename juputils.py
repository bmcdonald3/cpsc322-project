import random
from tabulate import tabulate

def convert_to_rating(mpg_list):
    for i in range(len(mpg_list)):
        mpg_list[i] = get_rating(mpg_list[i])
    return mpg_list
        
def get_rating(mpg):
    if mpg < 14:
        return 1
    elif mpg < 15:
        return 2
    elif mpg < 17:
        return 3
    elif mpg < 20:
        return 4
    elif mpg < 24:
        return 5
    elif mpg < 27:
        return 6
    elif mpg < 31:
        return 7
    elif mpg < 37:
        return 8
    elif mpg < 45:
        return 9
    return 10

def convert_weight(weight):
    res = []
    for val in weight:
        res.append(get_weight(val))
    return res

def get_weight(val):
    if val < 2000:
        curr = 1
    elif val < 2500:
        curr = 2
    elif val < 3000:
        curr = 3
    elif val < 3500:
        curr = 4
    else:
        curr = 5
    return curr

def get_rand_rows(table, num_rows):
    rand_rows = []
    for i in range(num_rows):
        rand_rows.append(table.data[random.randint(0,len(table.data))-1])
    return rand_rows

def print_pred_actual(rows, actual, predicted):
    for i in range(len(rows)):
        print('instance:', rows[i])
        print('class:', predicted[i], 'actual:', actual[i])

def get_accuracy(actual, predicted):
    predicted_correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            predicted_correct+=1
    return predicted_correct/len(actual)

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

def print_tabulate(table, headers):
    print(tabulate(table, headers, tablefmt="rst"))

def add_conf_stats(matrix):
    del matrix[0]
    for i,row in enumerate(matrix):
        row[0] = i+1
        row.append(sum(row))
        row.append(round(row[i+1]/row[-1]*100,2))
        
def titanic_stats(matrix):
    for i,row in enumerate(matrix):
        row.append(sum(row))
        row.append(round(row[i]/row[-1]*100,2))
        row.insert(0, i+1)
    matrix.append(['Total', matrix[0][1]+matrix[1][1], matrix[0][2]+matrix[1][2], matrix[0][3]+matrix[1][3], \
                   round(((matrix[0][1]+matrix[1][2])/(matrix[0][3]+matrix[1][3])*100),2)])
    