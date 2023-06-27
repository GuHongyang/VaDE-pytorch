import argparse
from dataloader import *
from model import VaDE
from tqdm import tqdm
import numpy as np
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import torch.nn as nn
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    w2 = 0
    for i in range(D):
        w2 += w[ind[0][i], ind[1][i]]
    w2 = w2 *1.0 / Y_pred.size
    return w2, w




if __name__ == '__main__':


    parse=argparse.ArgumentParser(description='VaDE')
    parse.add_argument('--batch_size',type=int,default=800)
    parse.add_argument('--datadir',type=str,default='./data/mnist')
    parse.add_argument('--nClusters',type=int,default=10)

    parse.add_argument('--hid_dim',type=int,default=10)
    parse.add_argument('--cuda',type=bool,default=True)


    args=parse.parse_args()



    DL,_=get_mnist(args.datadir,args.batch_size)

    vade=VaDE(args)
    if args.cuda:
        vade=vade.cuda()
        #vade=nn.DataParallel(vade,device_ids=range(4))

    vade.pre_train(DL,pre_epoch=50)


    opti=Adam(vade.parameters(),lr=2e-3)
    lr_s=StepLR(opti,step_size=10,gamma=0.95)


    writer=SummaryWriter('./logs')

    epoch_bar=tqdm(range(300))

    tsne=TSNE()

    for epoch in epoch_bar:

        lr_s.step()
        L=0
        for x,_ in DL:
            if args.cuda:
                x=x.cuda()

            loss=vade.ELBO_Loss(x)

            opti.zero_grad()
            loss.backward()
            opti.step()

            L+=loss.detach().cpu().numpy()


        pre=[]
        tru=[]

        with torch.no_grad():
            for x, y in DL:
                if args.cuda:
                    x = x.cuda()

                tru.append(y.numpy())
                pre.append(vade.predict(x))


        tru=np.concatenate(tru,0)
        pre=np.concatenate(pre,0)


        writer.add_scalar('loss',L/len(DL),epoch)
        writer.add_scalar('acc',cluster_acc(pre,tru)[0]*100,epoch)
        writer.add_scalar('lr',lr_s.get_lr()[0],epoch)

        epoch_bar.write('Loss={:.4f},ACC={:.4f}%,LR={:.4f}'.format(L/len(DL),cluster_acc(pre,tru)[0]*100,lr_s.get_lr()[0]))








