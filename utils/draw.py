
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# print(plt.style.available)
"""
['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'] 
"""
    
def drawplt_line(x, y, xlabel, ylabel, title, savepth):

    plt.style.use('seaborn-v0_8') 
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3.5), tight_layout=True, dpi=500) 

    sns_plot=sns.lineplot(x=x, y=y, marker='o', legend=False)
    ax.set_xlabel(xlabel, fontsize=12)    
    ax.set_ylabel(ylabel, fontsize=12)
    
    if len(x)<=20:
        plt.xticks(range(1, len(x) + 1), range(1, len(x) + 1), rotation=45)
    elif len(x)>20 and len(x)<=40:
        plt.xticks(range(1, len(x) + 1, 2), range(1, len(x) + 1, 2), rotation=45)
    elif len(x)>40 and len(x)<=60:
        plt.xticks(range(1, len(x) + 1, 3), range(1, len(x) + 1, 3), rotation=45)
    elif len(x)>60 and len(x)<=80:
        plt.xticks(range(1, len(x) + 1, 4), range(1, len(x) + 1, 4), rotation=45)
    elif len(x)>80 and len(x)<=100:
        plt.xticks(range(1, len(x) + 1, 5), range(1, len(x) + 1, 5), rotation=45)
    # plt.legend(title=None, frameon=False)
    ax.set_title(title,fontsize=12); 

    plt.tight_layout()
    plt.savefig(fname=f'{savepth}/{ylabel}.png',dpi=500)
    plt.close   