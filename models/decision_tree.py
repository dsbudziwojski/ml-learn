from models.utilities import tree, functions


def decision_tree(features, X, Y):
    def decision_tree_rec(features, X,Y):
        guess, num = functions.mostFrequentLabel(features, Y)
        if num == len(Y):
            return tree.Node(guess)
        elif len(features) == 0:
            return tree.Node(guess)
        else:
            # splitting
            f = functions.bestFeature(features, Y)
            new_features = [i for i in features if features != f]
            NO_x,NO_y, YES_x, YES_y = functions.splitData(features, X, Y)
            left = decision_tree_rec(new_features, NO_x, NO_y)
            right = decision_tree_rec(new_features, YES_x, YES_y)
            return tree.Node(f, left, right)

    return decision_tree_rec(features, X, Y)


