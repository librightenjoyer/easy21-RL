from env import Easy21
import numpy as np
import pickle

actions = [0, 1]
gamma = 1
N0 = 100

env = Easy21()

nPlayerSum = env.gameUpperBound + 1
nDeallerShownCard = env.maxCardValue + 1

Q = np.zeros((nPlayerSum, nDeallerShownCard, len(actions)))
NSA = np.zeros((nPlayerSum, nDeallerShownCard, len(actions)))

NS = lambda pv, dv: np.sum(NSA[pv, dv])
alpha = lambda pv, dv, act: 1/NSA[pv, dv, act]
epsilon = lambda pv, dv: N0 / (N0 + NS(pv, dv))

def epsilonGreedy(pv, dv):
    if np.random.random() < epsilon(pv, dv):
        action = np.random.choice(actions)
    else:
        action = np.argmax([Q[pv, dv, act] for act in actions])
    return action

episodes = int(1e7)
meanReturn = 0
wins = 0

for episode in range(episodes):
    SAR = []
    pv, dv = env.beginGame()
    
    terminated = False
    while not terminated:
        act = epsilonGreedy(pv, dv)
        NSA[pv, dv, act] += 1

        pNew, dNew, r, terminated = env.step(pv, dv, act)

        SAR.append([pv, dv, act, r])
        pv, dv = pNew, dNew

    G = sum([sar[-1] for sar in SAR])
    for (pv, dv, act, r) in SAR:
        Q[pv, dv, act] += alpha(pv, dv, act) * (G - Q[pv, dv, act])

    meanReturn = meanReturn + 1/(episode+1) * (G - meanReturn)
    if ((episode < 1e6) and (episode % 1e5 == 0)) or episode % 1e6 == 0:
        print("Episode %i, Mean Return %.3f"%(episode, meanReturn))

pickle.dump(Q, open('data/Q_table.dill', 'wb'))

