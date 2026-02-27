import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_theme()

def plot(Q, actions):
    pRange = list(range(1, 22))
    dRange = list(range(1, 11))
    vStar = []
    
    for p in pRange:
        for d in dRange:
            vStar.append([p, d, np.max([Q[p, d, a] for a in actions])])

    df = pd.DataFrame(vStar, columns=['player', 'dealer', 'value'])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    plt.show()

    surf = ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    ax.view_init(30, 45)
    plt.show()

    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.jet, linewidth=0.01)
    plt.show()

def plotMseEpisodesLambdas(arr):
    m, n = arr.shape
    I, J = np.ogrid[:m, :n]
    out = np.empty((m, n, 3), dtype=arr.dtype)
    out[..., 0] = I
    out[..., 1] = J
    out[..., 2] = arr
    out.shape = (-1, 3)

    df = pd.DataFrame(out, columns=['lambda', 'Episode', 'MSE'])
    df['lambda'] = df['lambda'] / 10
    
    g = sns.FacetGrid(df, hue="lambda", height=8, legend_out=True)
    g = g.map(plt.plot, "Episode", "MSE").add_legend()

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('MSE/Episode')
    plt.show()

def plotMseLambdas(data, lambdas):
    df = pd.DataFrame(data, columns=['MSE'])
    df['lambda'] = lambdas

    sns.pointplot(x=df['lambda'], y=df['MSE'])
    plt.title("MSE/Lambda")
    plt.show()