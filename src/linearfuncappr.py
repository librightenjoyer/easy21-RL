from env import Easy21
import numpy as np
import pickle

env = Easy21()
N0 = 100
actions = [0, 1]

trueQ = pickle.load(open('data/Q_table.dill', 'rb'))

alpha = 0.01
epsilon = 0.05

episodes = int(1e4)
lmds = list(np.arange(0,11)/10)
mseLambdas = np.zeros((len(lmds), episodes))
finalMSE = np.zeros(len(lmds))

def epsilonGreedy(pv, dv):
    if np.random.random() < epsilon:
        action = np.random.choice(actions)
    else:
        action = np.argmax([Q(pv, dv, act) for act in actions])
    return action

def allQ():
    return np.dot(allFeatures.reshape(-1, 3*6*2), theta).reshape(-1)

def features(pv, dv, act):
    f = np.zeros(3*6*2)

    for fi, (lower, upper) in enumerate(zip(range(1,8,3), range(4, 11, 3))):
        f[fi] = (lower <= dv <= upper)

    for fi, (lower, upper) in enumerate(zip(range(1,17,3), range(6, 22, 3)), start=3):
        f[fi] = (lower <= pv <= upper)

    f[-2] = 1 if act == 0 else 0
    f[-1] = 1 if act == 1 else 0
    return f.reshape(1, -1)

def Q(pv, dv, act):
    return np.dot(features(pv,dv,act), theta)

allFeatures = np.zeros((22, 11, 2, 3*6*2))

for pv in range(1, 22):
    for dv in range(1, 11):
        for act in range(0, 2):
            allFeatures[pv-1, dv-1, act] = features(pv, dv, act)

for li, lmd in enumerate(lmds):
    theta = np.random.randn(3*6*2, 1)

    for episode in range(episodes):
        terminated = False
        E = np.zeros_like(theta)

        pv, dv = env.beginGame()
        act = epsilonGreedy(pv, dv)

        while not terminated:
            pNew, dNew, r, terminated = env.step(pv, dv, act)

            if not terminated:
                actNew = epsilonGreedy(pNew, dNew)
                tdError = r + Q(pNew, dNew, actNew) - Q(pv, dv, act)
            else:
                tdError = r - Q(pv, dv, act)

            E = lmd * E + features(pv, dv, act).reshape(-1, 1)
            gradient = alpha * tdError * E
            theta = theta + gradient

            if not terminated:
                pv, dv, act = pNew, dNew, actNew

        mse = np.sum(np.square(allQ() - trueQ.ravel())) / (21*10*2)
        mseLambdas[li, episode] = mse

        if episode % 1000 == 0 or episode+1==episodes:
            print("Lambda=%.1f Episode %04d, MSE %5.3f"%(lmd, episode, mse))

    finalMSE[li] = mse

pickle.dump([finalMSE, lmds, mseLambdas], open('data/postLFAProcessing.dill', 'wb'))
