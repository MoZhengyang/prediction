import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch

# 读取数据
mae_lstm = []
mae_re_lstm = []
mae_ss_lstm = []
for seed in range(100):
    try:
        mae_lstm.append(torch.load(f"loss/MAE_LSTM_seed{seed}.pt"))
        mae_re_lstm.append(torch.load(f"loss/MAE_LSTM_re_seed{seed}.pt"))
        mae_ss_lstm.append(torch.load(f"loss/MAE_LSTM_ss_seed{seed}.pt"))
    except:
        pass
mae_lstm = torch.cat(mae_lstm).reshape(-1, 15)
mae_re_lstm = torch.cat(mae_re_lstm).reshape(-1, 15)
mae_ss_lstm = torch.cat(mae_ss_lstm).reshape(-1, 15)
index = [0, 4, 9, 14]
mae_lstm = mae_lstm[:, index]
mae_re_lstm = mae_re_lstm[:, index]
mae_ss_lstm = mae_ss_lstm[:, index]

# 第一个Horizon
mae1 = torch.stack([mae_lstm[:, 0], mae_ss_lstm[:, 0], mae_re_lstm[:, 0]], 0)
mae1 = np.array(mae1.reshape(-1))
maex = np.array([e for e in [1, 2, 3] for i in range(10)]).reshape(-1)
data = pd.DataFrame({
    'mae1':mae1,
    'maex':maex
})

data = pd.DataFrame({
    'mae1':np.tile(np.arange(0, 100), 3)/40,
    'maex':np.repeat(np.arange(1, 4), 100)
})

config = {
"mathtext.fontset":'stix',
"xtick.direction":"in",
"xtick.major.size": 6,
"ytick.direction":"in",
"ytick.major.size": 6,
"font.serif": ['SimSun'],
"font.family":"times new roman",
'legend.frameon':False,
}

# Apply the default theme
sns.set_theme(context='talk', style="ticks", palette=sns.color_palette("deep"), font_scale=1.4, rc=config)
fig, ax = plt.subplots(dpi=160, figsize=(5, 4))
# sns.stripplot(x='maex', y='mae1', data=data, hue='maex', palette=sns.color_palette("muted")[0:3],
#              alpha=.5, zorder=4, size=7, legend=False, ax=ax)
g = sns.boxplot(x='maex', y='mae1', data=data, hue='maex', linewidth=4, ax=ax)
g.legend([])
g.axes.xaxis.set_ticks([])
g.set(xlabel="Horizon 1", ylabel="MAE")
g.set_xlim([0.2, 1.8])
g.set_ylim([-7, 5])
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()