import torch
import torch.nn as nn
import torch.optim as optim
from .common import MLP, ResNet18

# Auxiliary functions from A-GEM

def compute_offsets(task, nc_per_task, is_cifar):
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def store_grad(pp, grads, grad_dims, tid):
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2average(gradient, average_gradient):
    dotp = torch.dot(gradient.view(-1), average_gradient.view(-1))
    if dotp < 0:
        gradient.copy_(average_gradient)

# Combined Net Class

class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda
        self.reg = args.memory_strength  # Initialize EWC regularization strength

        # A-GEM memory
        self.memory_data = torch.FloatTensor(n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        self.grad_dims = [param.data.numel() for param in self.parameters()]
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # EWC memory
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None

        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        self.n_outputs = n_outputs
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        # A-GEM memory update
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # EWC memory update
        if t != self.current_task:
            self.net.zero_grad()
            if self.is_cifar:
                offset1, offset2 = compute_offsets(self.current_task, self.nc_per_task, self.is_cifar)
                self.ce(self.net(self.memx)[:, offset1: offset2], self.memy - offset1).backward()
            else:
                self.ce(self(self.memx, self.current_task), self.memy).backward()
            self.fisher[self.current_task] = []
            self.optpar[self.current_task] = []
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher[self.current_task].append(pg)
            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        # Compute gradient on previous tasks for A-GEM
        if len(self.observed_tasks) > 1:
            average_gradient = torch.zeros_like(self.grads[:, 0])
            if self.gpu:
                average_gradient = average_gradient.cuda()
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                past_task = self.observed_tasks[tt]
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task, self.is_cifar)
                ptloss = self.ce(self.forward(self.memory_data[past_task], past_task)[:, offset1: offset2], self.memory_labs[past_task] - offset1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)
                average_gradient += self.grads[:, past_task]
            average_gradient /= (len(self.observed_tasks) - 1)

        # Now compute the grad on the current minibatch
        self.zero_grad()
        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)

        # EWC regularization
        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * self.fisher[tt][i]
                l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()
        loss.backward()

        # A-GEM constraint checking and gradient projection
        if len(self.observed_tasks) > 1:
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            project2average(self.grads[:, t], average_gradient)
            overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)

        self.opt.step()
