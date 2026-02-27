from env import Easy21
import numpy as np
import pickle

env = Easy21()
N0 = 100
actions = [0, 1]

def reset():
    Q = np.zeros((22, 11, len(actions)))
    NSA = np.zeros((22, 11, len(actions)))
    wins = 0

    return Q, NSA, wins

Q, NSA, wins = reset()
trueQ = pickle.load(open('data/Q_table.dill', 'rb'))

NS = lambda pv, dv: np.sum(NSA[pv, dv])
alpha = lambda pv, dv, act: 1/NSA[pv, dv, act]
epsilon = lambda pv, dv: N0 / (N0 + NS(pv, dv))

def epsilonGreedy(pv, dv):
    if np.random.random() < epsilon(pv, dv):
        action = np.random.choice(actions)
    else:
        action = np.argmax([Q[pv, dv, act] for act in actions])
    return action

episodes = int(1e4)
lmds = list(np.arange(0,11)/10)
mselambdas = np.zeros((len(lmds), episodes))
finalMSE = np.zeros(len(lmds))

for li, lmd in enumerate(lmds):
    Q, NSA, wins = reset()

    for episode in range(episodes):
        terminated = False
        E = np.zeros((22, 11, len(actions)))
        pv, dv = env.beginGame()
        act = epsilonGreedy(pv, dv)
        SA = list()

        while not terminated:
            pNew, dNew, r, terminated = env.step(pv, dv, act)

            if not terminated:
                actNew = epsilonGreedy(pNew, dNew)
                tdError = r + Q[pNew, dNew, actNew] - Q[pv, dv, act]
            else:
                tdError = r - Q[pv, dv, act]

            E[pv, dv, act] += 1
            NSA[pv, dv, act] += 1
            SA.append([pv, dv, act])

            for (_pv, _dv, _act) in SA:
                Q[_pv, _dv, _act] += alpha(_pv, _dv, _act) * tdError * E[_pv, _dv, _act]
                E[_pv, _dv, _act] *= lmd
            if not terminated:
                pv, dv, act = pNew, dNew, actNew

        mse = np.sum(np.square(Q-trueQ)) / (21*10*2)

        mselambdas[li, episode] = mse

        if episode % 1000 == 0 or episode + 1 == episodes:
            print("Lambda=%.1f Episode %04d, MSE %5.3f"%(lmd, episode, mse))
    finalMSE[li] = mse

pickle.dump([finalMSE, lmds, mselambdas], open('data/postTDProcessing.dill', 'wb'))
