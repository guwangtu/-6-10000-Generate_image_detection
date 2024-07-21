import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim 
  
from .diff_denoise import purify

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                x_adv=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    if x_adv == None:
        model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        elif distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
def mart_loss(model,
              x_natural,
              y,
              optimizer,
              x_adv=None,              
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
     
    batch_size = len(x_natural)
    # generate adversarial example
    if x_adv == None:
        model.eval()
        device=next(model.parameters()).device
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_ce = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss


def diff_denoise_loss(
                train_model,
                df_model,
                diffusion,
                y,
                criterion,
                x_adv,
                t):
     
    adv_x_denoise=purify(x_adv, t, df_model, diffusion)
    pred_y= train_model(adv_x_denoise)  
    return criterion(pred_y,y)


