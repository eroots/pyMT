import e_colours.colourmaps as cm
import matplotlib.pyplot as plt
import numpy as np


N = 64
save_path = 'C:/Users/eroots/phd/ownCloud/Documents/common_figures/colour_bars/'
for colour_map in cm.COLOUR_MAPS:
    cmap = cm.get_cmap(colour_map, N)
# plt.figure()
# sc = plt.scatter(range(10), range(10), c=range(10), cmap=cmap)

    fig, ax = plt.subplots(1, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax.imshow([cmap(np.arange(N))], extent=[0, 10, 0, 1])
    plt.savefig(save_path + colour_map, dpi=300)
# plt.show()