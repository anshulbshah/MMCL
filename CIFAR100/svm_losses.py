from __future__ import print_function
import torch
import torch.nn as nn
import time
import torch.autograd
from torch.autograd import Variable
import numpy as np
from solvers import *

def compute_kernel_new(X,Y,gamma=0.1):
    gamma = 1./float(gamma)
    distances = -gamma*(2-2.*torch.mm(X,Y.T))
    kernel = torch.exp(distances)
    return kernel

class MMCL_inv(nn.Module):
    def __init__(self, sigma=0.07, batch_size=256, anchor_count=2, C=1.0):
        super(MMCL_inv, self).__init__()
        self.sigma = sigma
        self.C = C
        
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

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*0.1).cuda()
        self.one_bs = torch.ones(batch_size, nn, 1).cuda()
        self.one = torch.ones(nn,).cuda()
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().cuda(); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().cuda(); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().cuda()
        
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
        
    def forward(self, features, labels=None, mask=None):
        bs = features.shape[0]
        nn = bs - 1
        
        F = torch.cat(torch.unbind(features, dim=1), dim=0)
        K = compute_kernel_new(F[:nn+1], F, gamma=self.sigma)
        
        
        with torch.no_grad():
            KK = torch.masked_select(K.detach(), self.block).reshape(bs, bs)
        
            KK_d0 = KK*self.no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1,bs,1)
            KXY = KXY + KXY.transpose(2,1)                
            Delta = (self.oneone + KK).unsqueeze(0) + KXY
            
            DD = torch.masked_select(Delta, self.KMASK).reshape(bs, nn, nn)
            
            alpha_y, _ = torch.solve(2*self.one_bs, DD)
            alpha_y = alpha_y.squeeze(2)

            if self.C == -1:
                alpha_y = torch.relu(alpha_y).detach()
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()

            alpha_x = alpha_y.sum(1)
            
        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T
        
        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()/bs
        loss = neg_loss - pos_loss

        return -pos_loss, neg_loss


class MMCL_pgd(nn.Module):
    def __init__(self, sigma=0.07, batch_size=256, anchor_count=2, C=1.0, num_iter=1000, eta=1E-3, stop_condition=0.01, solver_type='nesterov', use_norm='false'):
        super(MMCL_pgd, self).__init__()
        self.sigma = sigma
        self.C = C
        
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

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*0.1).cuda()
        self.one_bs = torch.ones(batch_size, nn, 1).cuda()
        self.one = torch.ones(nn,).cuda()
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().cuda(); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().cuda(); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().cuda()
        self.num_iters = num_iter
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
        
    def forward(self, features, labels=None, mask=None):

        bs = features.shape[0]
        nn = bs - 1
        
        F = torch.cat(torch.unbind(features, dim=1), dim=0)
        K = compute_kernel_new(F[:nn+1], F, gamma=self.sigma)
        

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
                alpha_y = torch.relu(alpha_y).detach()
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()
            alpha_x = alpha_y.sum(1)
        

        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T
        
        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()/bs
        loss = neg_loss - pos_loss

        return -pos_loss, neg_loss


