import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier as sk_DecisionTreeClassifier

class DecisionTreeClassifier(sk_DecisionTreeClassifier):

    def __init__(self, criterion = 'gini', splitter = 'best', max_depth = None, min_samples_split = 2,
    min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_features = None, random_state = None,
     max_leaf_nodes = None, min_impurity_decrease = 0.0, class_weight = None, ccp_alpha = 0.0):

     super(DecisionTreeClassifier, self).__init__(criterion = criterion, splitter = splitter, 
        max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf,
        min_weight_fraction_leaf = min_weight_fraction_leaf, max_features = max_features, 
        random_state = random_state, max_leaf_nodes = max_leaf_nodes, 
        min_impurity_decrease = min_impurity_decrease, class_weight = class_weight, 
        ccp_alpha = ccp_alpha)

    def tree_explainer(self):
        
        tree = self.tree_

        tree_data = {'node': [],
                    'left': [],
                    'right': [],
                    'feature': [],
                    'threshold': []}

        for i in range(tree.node_count):
            tree_data['node'].append(i)
            tree_data['left'].append(tree.children_left[i])
            tree_data['right'].append(tree.children_right[i])

            if tree.children_left[i] != -1 and tree.children_right[i] != -1:
                tree_data['feature'].append(tree.feature[i])
                tree_data['threshold'].append(tree.threshold[i])
            else:
                tree_data['feature'].append(None)
                tree_data['threshold'].append(None)

        return pd.DataFrame(tree_data)

if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame = True)
    data = iris['data']
    target = iris['target']
    
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X = data, y = target)
    explainer = decision_tree_model.tree_explainer()

    print(explainer)