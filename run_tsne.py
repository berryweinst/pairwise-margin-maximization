import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from random import randint, shuffle
import os

target_idx = list(range(10))
# shuffle(target_idx)


# while len(set(target_idx)) != 10*10:
#     target_idx = list(range(100))
#     shuffle(target_idx)
#     target_idx = [34, 98, 99, 49, 37, 88, 60, 72, 83] + target_idx[0:6] + [44, 19, 2, 89, 40, 55, 3, 47, 46, 94, 43, 78] + target_idx[6:9]



# colors = []
# for i in range(10):
#     colors.append('#%06X' % np.arange(100, 0xFFFFFF, 100))


def get_cmap(n, name='tab10'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
#
cmap = get_cmap(10)

tsne = TSNE(n_components=2, random_state=0)

# if os.path.isfile('./baseline_X_2d.npy'):
#     baseline_X_2d = np.load('./baseline_X_2d.npy')
#     pmm_X_2d = np.load('./pmm_X_2d.npy')
# else:
baseline_features = np.load('./cifar10_feature_maps_baseline.npy')
pmm_features = np.load('./cifar10_feature_maps_pmm.npy')

# baseline_X_2d = tsne.fit_transform(baseline_features)
# np.save('./baseline_X_2d', baseline_X_2d)
# pmm_X_2d = tsne.fit_transform(pmm_features)
# np.save('./pmm_X_2d', pmm_X_2d)
target = np.load('./cifar10_targets.npy')


print("Done calculating TSNE")


fig1, axes1 = plt.subplots(nrows=1,ncols=2, figsize=(30,15), sharex=True, sharey=True)
# fig1.subplots_adjust(hspace=.2,wspace=0.1)

baseline_counter = 0
pmm_counter = 0

for fidx, ax in enumerate(axes1.flatten()[:1]):
    all_idx = np.sum([target == id for id in target_idx[fidx*10: (fidx + 1)*10]], axis=0)
    baseline_X_2d = tsne.fit_transform(baseline_features[all_idx!=0])
    ax.set_title('classes' + ','.join(str(e) for e in target_idx[fidx*10: (fidx + 1)*10]))
    for i, id in enumerate(target_idx[fidx*10: (fidx + 1)*10]):
        s = 100
        ax.scatter(baseline_X_2d[target[all_idx!=0] == id, 0], baseline_X_2d[target[all_idx!=0] == id, 1], c=cmap(i), label=id, alpha=0.2, marker='+', s=s)
        baseline_counter += len(baseline_X_2d[target[all_idx!=0] == id, 0])
        # ax.legend()


for fidx, ax in enumerate(axes1.flatten()[1:]):
    all_idx = np.sum([target == id for id in target_idx[fidx * 10: (fidx + 1) * 10]], axis=0)
    pmm_X_2d = tsne.fit_transform(pmm_features[all_idx != 0])
    for i, id in enumerate(target_idx[fidx*10: (fidx + 1)*10]):
        s = 100
        ax.scatter(pmm_X_2d[target[all_idx!=0] == id, 0], pmm_X_2d[target[all_idx!=0] == id, 1], c=cmap(i), label=id, alpha=0.2, marker='+', s=s)
        last_ax = ax
        pmm_counter += len(pmm_X_2d[target[all_idx!=0] == id, 0])

# handles, labels = last_ax.get_legend_handles_labels()
# fig1.legend(handles, labels, loc='upper center')

print("Baseline count = %d" % (baseline_counter))
print("PMM count = %d" % (pmm_counter))

plt.tight_layout()
plt.show()