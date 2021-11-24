import numpy as np
import random
import torch


def xyLocs(mask):
    y, x = mask.nonzero()
    return list(zip(x, y))


def actionMask(actions, maxActions=573):
    return np.isin([n for n in range(maxActions)], actions).astype(int)


def epsilonGreedyActionSelect(actionScores, availableActions, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return np.random.choice(availableActions), True
    else:
        mask = actionMask(availableActions)
        filteredScores = [actionScores[i] for i in range(len(mask)) if mask[i] != 0]
        maxScore = max(filteredScores)
        maxIndex = filteredScores.index(maxScore)
        return availableActions[maxIndex], False


def argSelect(argScores, randomSelect):
    if randomSelect:
        shape = np.shape(argScores)
        selectedIndex = []
        for dim in shape:
            selectedIndex.append(np.random.randint(0, dim))
        return selectedIndex
    else:
        # if len(np.shape(argScores)) > 1:
        #     print(np.array([np.argmax(argScores)]))
        #     return np.array([np.argmax(argScores)])
        # else:
        #     print(np.array(np.argmax(argScores)))
        #     return np.array(np.argmax(argScores))
        return np.array(np.unravel_index(np.argmax(argScores), np.shape(argScores)))


def calculateDiscountedScoreDeltas(scores, discountFactor=0.95):
    deltas = [0]
    for i in range(1, len(scores)):
        deltas.append(scores[i] - scores[i-1])
    discountedScores = [deltas[-1]]
    for i in range(1, len(scores)):
        ds = discountedScores[i - 1] * discountFactor + deltas[-(i + 1)]
        discountedScores.append(ds)
    discountedScores.reverse()
    return discountedScores


def policyActionSelect(policy, availableActions):
    mask = actionMask(availableActions)
    flat = np.ravel(policy)

    maskedPolicy = np.multiply(mask, flat)
    maskedSum = np.sum(maskedPolicy)
    if maskedSum == 0:
        # print("Random action taken")
        return np.random.choice(availableActions)
    maskedPolicy = np.divide(maskedPolicy, maskedSum)
    # print(maskedPolicy)

    indices = np.arange(len(maskedPolicy))

    # actionProbabilities = [n for n in maskedPolicy if n > 0.01]
    # actionProbabilities.sort(reverse=True)
    # print(actionProbabilities)

    index = np.random.choice(indices, p=maskedPolicy)
    return index


def policyArgSelect(policy):
    flat = np.ravel(policy)
    indices = np.arange(len(flat))
    index = np.random.choice(indices, p=flat)
    return np.array(np.unravel_index(index, np.shape(policy)))


def convertToOneHot(categoricalValue, numCategories):
    output = np.zeros(numCategories)
    output[categoricalValue] = 1
    return output


def convertToOneHot2D(categorical2D, numCategories):
    shape = (numCategories,) + np.shape(categorical2D)
    output = np.zeros(shape)
    for i in range(len(categorical2D)):
        for j in range(len(categorical2D[i])):
            output[categorical2D[i][j]][i][j] = 1

    return output


def convertToTensors(arrays, unsqueeze=True):
    tensors = []
    for a in arrays:
        if unsqueeze:
            t = torch.unsqueeze(torch.tensor(a).float().cuda(), 0)
        else:
            t = torch.tensor(a).float().cuda()
        tensors.append(t)
    return tensors


def detachAllTensors(tensors):
    detached = []
    for t in tensors:
        detached.append(t.detach().clone().cpu())
    return detached


def moveAllToCuda(tensors):
    cuda = []
    for t in tensors:
        cuda.append(t.cuda())
    return cuda