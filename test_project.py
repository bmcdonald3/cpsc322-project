import numpy as np
import scipy.stats as stats 
import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyRandomForestClassifier, MyDecisionTreeClassifier
import operator
import random

def test_decision_tree_classifier_fit():
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    interview_tree = \
        ['Attribute', 'att0', 
            ['Value', 'Senior', 
                ['Attribute', 'att2', 
                    ['Value', 'no', 
                        ['Leaf', 'False', 3, 5]
                    ], 
                    ['Value', 'yes', 
                        ['Leaf', 'True', 2, 5]
                    ]
                ]
            ], 
            ['Value', 'Mid', 
                ['Leaf', 'True', 4, 14]
            ], 
            ['Value', 'Junior', 
                ['Attribute', 'att3', 
                    ['Value', 'no', 
                        ['Leaf', 'True', 3, 5]
                    ], 
                    ['Value', 'yes', 
                        ['Leaf', 'False', 2, 5]
                    ]
                ]
            ]
        ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    myt = MyDecisionTreeClassifier()
    myt.fit(X_train, y_train)
    assert(myt.tree == interview_tree)

    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]
    degrees_y_train = []
    for row in degrees_table:
        degrees_y_train.append(row[-1])
        del row[-1]
    
    myt.fit(degrees_table, degrees_y_train)

    degrees_tree = \
        ['Attribute', 'att0', 
            ['Value', 'A', 
                ['Attribute', 'att4', 
                    ['Value', 'B', 
                        ['Attribute', 'att3', 
                            ['Value', 'B', 
                                ['Leaf', 'SECOND', 7, 9]
                            ], 
                            ['Value', 'A', 
                                ['Attribute', 'att1', 
                                    ['Value', 'B', 
                                        ['Leaf', 'SECOND', 1, 2]
                                    ], 
                                    ['Value', 'A', 
                                        ['Leaf', 'FIRST', 1, 2]
                                    ]
                                ]
                            ]
                        ]
                    ], 
                    ['Value', 'A', 
                        ['Leaf', 'FIRST', 5, 14]
                    ]
                ]
            ], 
            ['Value', 'B', 
                ['Leaf', 'SECOND', 12, 26]
            ]
        ]    

    assert(myt.tree == degrees_tree)

def test_simple_linear_regressor_fit():
    myline = MyRandomForestClassifier(2, 5, 3)
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    y_domain = myutils.get_unique(y_train)
    myline.fit(X_train, y_train)
    prediction = myline.predict([["Junior", "Python", "no", "yes"], ["Mid", "Java", "yes", "no"]])
    for val in prediction:
        assert(val in y_domain)
