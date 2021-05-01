from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from options import args_parser
import torch
import random

def visualize(args, x, y):
    fig = plt.figure(figsize=(4, 3))
    print("Begin visualization ...")
    markers = ['.', 'o', 'v', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # colors = ['#C0C0C0', 'lightcoral', 'bisque', 'lemonchiffon', 'lightcyan', 'lavender', 'yellowgreen', 'lavenderblush', 'thistle', 'aquamarine']
    # colors_1 = ['#000000', '#FF0000', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'sage', 'palevioletred', 'darkviolet', 'g']
    colors = ['#000000', 'peru', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'darkseagreen', 'violet', 'palevioletred', 'g']

    Label_Com = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 14
             }

    # S_data = np.hstack((x, y, d))
    S_data = np.hstack((x, y))
    # S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2], 'device': S_data[:,3]})
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    # for class_index in range(args.num_classes):
    #     for device_index in range(args.num_users):
    #         X = S_data.loc[(S_data['label'] == class_index) & (S_data['device'] == device_index)]['x']
    #         Y = S_data.loc[(S_data['label'] == class_index) & (S_data['device'] == device_index)]['y']
    #         # plt.scatter(X, Y, cmap='brg', s=100, marker='.', c='', edgecolors=colors[index], alpha=0.65)
    #         # plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.6)
    #         plt.scatter(X, Y, marker=markers[device_index], c=colors[class_index])

    for class_index in range(args.num_classes):
        X = S_data.loc[S_data['label'] == class_index]['x']
        Y = S_data.loc[S_data['label'] == class_index]['y']
        # plt.scatter(X, Y, cmap='brg', s=100, marker='.', c='', edgecolors=colors[index], alpha=0.65)
        # alpha set: fedavg-0.1, fedsem-0.08, fedper-0.03, hetfel-0.03
        plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.08)


    # for class_index in range(args.num_classes):
    #     # global prototype
    #     X = S_data.loc[(S_data['label'] == class_index)]['x']
    #     Y = S_data.loc[(S_data['label'] == class_index)]['y']
    #     x_avg = np.average(X)
    #     y_avg = np.average(Y)
    #     plt.scatter(x_avg, y_avg, marker='o', c=colors[class_index], s=50)


    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值

    # plt.title(args.alg, fontsize=14, fontweight='normal', pad=20)
    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.1, hspace=0.15)
    # plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.4, columnspacing=0.4, markerscale=2,
    #            bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)
    # plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.4, columnspacing=0.4,
    #            bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)

    # fig.show()
    plt.savefig("./protos_"+args.alg+".pdf", format='pdf', dpi=600)

args = args_parser()
args.alg = 'fedper'

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device == 'cuda':
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

x = np.load('/Users/tanyue/Desktop/saved/protos/' + args.alg + '_protos.npy', allow_pickle=True)
y = np.load('/Users/tanyue/Desktop/saved/protos/' + args.alg + '_labels.npy', allow_pickle=True)
# d = np.load('../protos/' + args.alg + '_idx.npy', allow_pickle=True)

tsne = TSNE()
x = tsne.fit_transform(x)

# x = x[:,0:2]

y = y.reshape((-1, 1))
# d = d.reshape((-1, 1))
# visualize(args, x, y, d)
visualize(args, x, y)


# from mlxtend.plotting import plot_decision_regions
# from sklearn.svm import SVC
# from mlxtend.data import iris_data
# clf = SVC(random_state=0, probability=True)
# # X, y = iris_data()
# # X = X[:,[0, 2]]
# clf.fit(x, y)
# fig = plt.figure(figsize=(10, 8))
# fig = plot_decision_regions(X=x, y=y, clf=clf, legend=2)
# plt.title(args.alg+'mu100')
# plt.show()