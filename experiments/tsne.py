import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
from torch.utils.data import DataLoader

from neurve.tsne import FlatMNIST, MLP, TSNETrainer

dset = FlatMNIST(train=True, download=True, root="./data")
train_dl = DataLoader(dset, batch_size=500)
net = MLP(28 * 28)
opt = SGD(params=net.parameters(), lr=1.0)
trainer = TSNETrainer(
    perplexity=10, data_loader=train_dl, net=net, opt=opt, out_path="test"
)
trainer.train(10)

net.eval()
val_dset = FlatMNIST(train=False, download=True, root="./data", return_labels=True)
val_dl = DataLoader(val_dset, batch_size=500)

all_embs, all_labels = np.empty((0, 2)), []
for x, y in val_dl:
    all_embs = np.concatenate([all_embs, net(x).detach().cpu().numpy()])
    all_labels.extend(y.detach().cpu().numpy().tolist())

plt.scatter(all_embs[:, 0], all_embs[:, 1], c=all_labels, s=5)
plt.show()
