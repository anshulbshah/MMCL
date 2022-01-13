import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist
from models.solvers import *

def compute_kernel(X,Y,gamma=0.1, kernel_type='rbf'):
    if kernel_type == 'linear':
        kernel = torch.mm(X,Y.T)
    elif kernel_type == 'rbf':
        if gamma == 'auto':
            gamma = 1/X.shape[-1]
        gamma = 1./float(gamma)
        #distances = torch.cdist(X,Y)
        distances = -gamma*(2-2.*torch.mm(X,Y.T))
        kernel = torch.exp(distances)
        
    elif kernel_type == 'poly':
        kernel = torch.pow(torch.mm(X, Y.T)+0.5, 3.)
    elif kernel_type == 'tanh':
        kernel = torch.tanh(gamma*torch.mm(X,Y.T))
    elif kernel_type == 'min':
        #kernel = torch.minimum(torch.relu(X), torch.relu(Y))
        kernel = torch.min(torch.relu(X).unsqueeze(1), torch.relu(Y).unsqueeze(1).transpose(1,0)).sum(2)
      
    return kernel


def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg

def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    precision = (1 + torch.arange(k, device=logits.device).float()) / labels_to_sorted_idx
    return precision.sum(1) / k


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z, get_map=False):

        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])

            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        if get_map:
            _map = mean_average_precision(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)
            return loss, acc, _map

        return loss, acc


class MMCL_INV(nn.Module):
    def __init__(self, sigma=0.07, contrast_mode='all',
                 base_sigma=0.07, batch_size=256, anchor_count=2, C=1.0, kernel='rbf',reg=0.1, schedule=[], multiplier=2, distributed=False):
        super(MMCL_INV, self).__init__()
        self.sigma = sigma
        self.contrast_mode = contrast_mode
        self.base_sigma = base_sigma
        self.C = C
        self.kernel = kernel
        
        nn = batch_size - 1
        bs = batch_size
        
        self.mask, self.logits_mask = self.get_mask(batch_size, anchor_count)
        self.eye = torch.eye(anchor_count*batch_size).cuda()
        
        self.pos_mask = self.mask[:bs, bs:].bool()
        neg_mask=(self.mask*self.logits_mask+1)%2; 
        self.neg_mask = neg_mask-self.eye
        self.neg_mask = self.neg_mask[:bs, bs:].bool()
        
        self.kmask = torch.ones(batch_size,).bool().cuda()
        self.kmask.requires_grad = False
        self.reg = reg

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*reg).cuda()
        self.one_bs = torch.ones(batch_size, nn, 1).cuda()
        self.one = torch.ones(nn,).cuda()
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().cuda(); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().cuda(); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().cuda()
        self.bs = bs
        self.schedule = schedule
        self.multiplier = multiplier
        self.distributed = distributed

    def get_kmask(self, bs):
        KMASK = torch.ones(bs, bs, bs).bool().cuda()
        for t in range(bs):
            KMASK[t,t,:] = False
            KMASK[t,:,t] = False
        return KMASK.detach()
        
    def get_mask(self, batch_size, anchor_count):
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        return mask, logits_mask
        
    def forward(self, z):

        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            x_sorted = []
            z_sorted = []
            for x,z in zip(z_list[:-1:2],z_list[1::2]):
                x_sorted.append(x)
                z_sorted.append(z)

            x_sorted = torch.cat(x_sorted)
            z_sorted = torch.cat(z_sorted)

            ftr = torch.cat([x_sorted,z_sorted],0)


        bs = ftr.shape[0]//2
        nn = bs - 1
        K = compute_kernel(ftr[:nn+1], ftr, kernel_type=self.kernel, gamma=self.sigma)
        
        with torch.no_grad():
            KK = torch.masked_select(K.detach(), self.block).reshape(bs, bs)
        
            KK_d0 = KK*self.no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1,bs,1)
            KXY = KXY + KXY.transpose(2,1)
            Delta = (self.oneone + KK).unsqueeze(0) + KXY
    
            DD = torch.masked_select(Delta, self.KMASK).reshape(bs, nn, nn)
            
            alpha_y, _ = torch.solve(2*self.one_bs, DD)

            # if torch.rand(1)>0.99: 
            #     print('alpha_y.max=%f alpha_y.min=%f alpha_y.mean=%f: error=%f'%
            #           (alpha_y.max(), alpha_y.min(),alpha_y.mean(), (torch.bmm(DD, alpha_y)-2.*self.one_bs).norm()))
                
            alpha_y = alpha_y.squeeze(2)
            if self.C == -1:
                alpha_y = torch.relu(alpha_y)
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()
            alpha_x = alpha_y.sum(1)
            
        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T
        
        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()/bs
        loss = neg_loss - pos_loss

        num_zero = (alpha_y == 0).sum()/alpha_y.numel()
        sparsity = (alpha_y == self.C).sum()/((alpha_y>0).sum()+1e-10)
        return loss, (Ks*self.pos_mask).sum(1).mean(), Kn.mean(), sparsity, num_zero, 0.0


class MMCL_PGD(nn.Module):
    def __init__(self, sigma=0.07, contrast_mode='all',
                 base_sigma=0.07, batch_size=256, anchor_count=2, C=1.0, kernel='rbf',reg=0.1, schedule=[], multiplier=2, distributed=False, num_iters=1000, eta=1e-3, stop_condition=1e-2, solver_type='nesterov', use_norm='true'):
        super(MMCL_PGD, self).__init__()
        self.sigma = sigma
        self.contrast_mode = contrast_mode
        self.base_sigma = base_sigma
        self.C = C
        self.kernel = kernel
        
        nn = batch_size - 1
        bs = batch_size
        
        self.mask, self.logits_mask = self.get_mask(batch_size, anchor_count)
        self.eye = torch.eye(anchor_count*batch_size).cuda()
        
        self.pos_mask = self.mask[:bs, bs:].bool()
        neg_mask=(self.mask*self.logits_mask+1)%2; 
        self.neg_mask = neg_mask-self.eye
        self.neg_mask = self.neg_mask[:bs, bs:].bool()
        
        self.kmask = torch.ones(batch_size,).bool().cuda()
        self.kmask.requires_grad = False
        self.reg = reg

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*reg).cuda()
        self.one_bs = torch.ones(batch_size, nn, 1).cuda()
        self.one = torch.ones(nn,).cuda()
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().cuda(); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().cuda(); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().cuda()
        self.bs = bs
        self.schedule = schedule
        self.multiplier = multiplier
        self.distributed = distributed

        self.num_iters = num_iters
        self.eta = eta
        self.stop_condition = stop_condition
        self.solver_type = solver_type
        self.use_norm = False if use_norm=='false' else True

    def get_kmask(self, bs):
        KMASK = torch.ones(bs, bs, bs).bool().cuda()
        for t in range(bs):
            KMASK[t,t,:] = False
            KMASK[t,:,t] = False
        return KMASK.detach()
        
    def get_mask(self, batch_size, anchor_count):
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        return mask, logits_mask
        
    def forward(self, z):

        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug0>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # print('zlist len',len(z_list))
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            x_sorted = []
            z_sorted = []
        
            for x,z in zip(z_list[:-1:2],z_list[1::2]):
                x_sorted.append(x)
                z_sorted.append(z)

            x_sorted = torch.cat(x_sorted)
            z_sorted = torch.cat(z_sorted)

            ftr = torch.cat([x_sorted,z_sorted],0)


        bs = ftr.shape[0]//2
        nn = bs - 1

        K = compute_kernel(ftr[:nn+1], ftr, kernel_type=self.kernel, gamma=self.sigma)

        
        with torch.no_grad():
            KK = torch.masked_select(K.detach(), self.block).reshape(bs, bs)
        
            KK_d0 = KK*self.no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1,bs,1)
            KXY = KXY + KXY.transpose(2,1)
            Delta = (self.oneone + KK).unsqueeze(0) + KXY
    
            DD = torch.masked_select(Delta, self.KMASK).reshape(bs, nn, nn)

            if self.C == -1:
                alpha_y = torch.relu(torch.randn(bs,nn,1,device=DD.device))
            else:
                alpha_y = torch.relu(torch.randn(bs,nn,1,device=DD.device)).clamp(min=0, max=self.C)

            if self.solver_type == 'nesterov':
                alpha_y,iter_no,abs_rel_change,rel_change_init = pgd_with_nesterov(self.eta,self.num_iters,DD,2*self.one_bs,alpha_y.clone(),self.C,use_norm=self.use_norm,stop_condition=self.stop_condition)
            elif self.solver_type == 'vanilla':
                alpha_y,iter_no,abs_rel_change,rel_change_init = pgd_simple_short(self.eta,self.num_iters,DD,2*self.one_bs,alpha_y.clone(),self.C,use_norm=self.use_norm,stop_condition=self.stop_condition)
            
            alpha_y = alpha_y.squeeze(2)
            if self.C == -1:
                alpha_y = torch.relu(alpha_y)
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()
            alpha_x = alpha_y.sum(1)
                        
        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T
        
        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()/bs
        loss = neg_loss - pos_loss


        sparsity = (alpha_y == self.C).sum()/((alpha_y>0).sum()+1e-10)
        num_zero = (alpha_y == 0).sum()/alpha_y.numel()
        return loss, (Ks*self.pos_mask).sum(1).mean(), Kn.mean(), sparsity, num_zero, 0.0
