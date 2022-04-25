import torch
import numpy as np
from torch.autograd import Variable
import time


def pgd_simple_short(eta,num_iter,Q,p,alpha_y,C,use_norm=False,stop_condition=0.01):
    if use_norm:
        eta_to_use = 1.0/torch.norm(Q,dim=(1,2),p=2,keepdim=True)
    else:
        eta_to_use = eta
    theta1 = torch.eye(Q.shape[1],device=Q.device).unsqueeze(0).repeat(Q.shape[0],1,1) - eta_to_use*Q
    theta2 = eta_to_use*p
    rel_change_init = -1.0
    for iter_no in range(num_iter):

        x_new = torch.bmm(theta1,alpha_y) + theta2
        if C == -1:
            alpha_y_new = torch.relu(x_new).detach()
        else:
            alpha_y_new = torch.relu(x_new).clamp(min=0, max=C)

        abs_rel_change = ((alpha_y_new-alpha_y)/(alpha_y + 1E-7)).abs().mean()

        if iter_no == 0:
            rel_change_init = abs_rel_change
        if abs_rel_change < stop_condition:
            alpha_y = alpha_y_new
            break
        alpha_y = alpha_y_new


    return alpha_y,iter_no,abs_rel_change,rel_change_init

def pgd_with_nesterov(eta,num_iter,Q,p,alpha_y,C,use_norm=False,stop_condition=0.01):
    if use_norm:
        eta_to_use = 1.0/torch.norm(Q,dim=(1,2),p=2,keepdim=True)        
    else:
        eta_to_use = eta

    theta1 = torch.eye(Q.shape[1],device=Q.device).unsqueeze(0).repeat(Q.shape[0],1,1) - eta_to_use*Q
    theta2 = eta_to_use*p
    alpha0 = np.random.uniform(low=1E-8)
    y = alpha_y
    rel_change_init = -1.0
    for iter_no in range(num_iter):

        x_new = torch.bmm(theta1,y) + theta2

        if C == -1:
            alpha_y_new = torch.relu(x_new).detach()
        else:
            alpha_y_new = torch.relu(x_new).clamp(min=0, max=C)


        alpha0_new = 0.5*(np.sqrt(alpha0**4 + 4*alpha0**2) - alpha0**2)
        beta_k = (alpha0*(1-alpha0))/(alpha0**2 + alpha0_new)
        y = alpha_y_new + beta_k*(alpha_y_new - alpha_y)
        alpha0 = alpha0_new

        abs_rel_change = ((alpha_y_new-alpha_y)/(alpha_y + 1E-7)).abs().mean()
        if iter_no == 0:
            rel_change_init = abs_rel_change
        if abs_rel_change < stop_condition:
            alpha_y = alpha_y_new
            break
        alpha_y = alpha_y_new

    return alpha_y,iter_no,abs_rel_change,rel_change_init

