from math import log2

def entropy(l):
    """Calculates the entropy given a list of probabilities"""
    e = 0
    for i in l:
        e -= i*log2(i)
    return e

def mostFrequentLabel(features, Y):
    if len(features) == 0:
        return 0
    cur, curFrequency = features[0], 0
    for feature in features:
        tally = 0
        for item in Y:
            if feature == item:
                tally += 1
        if tally > curFrequency:
            curFrequency = tally
            cur = feature
    return cur, curFrequency


def bestFeature(features, X, Y):
    # TODO


def splitData(features, X, Y):
    # TODO