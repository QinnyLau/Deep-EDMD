import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Nets(nn.Module):
    def __init__(self, en0, en1, en2, en3, b0):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(en0, en1),
                                     nn.Linear(en1, en2),
                                     nn.Linear(en2, en3),)
        self.A = nn.Linear(en3, en3)
        self.B = nn.Linear(b0, en3)
        self.decoder = nn.Sequential(nn.Linear(en3, en2),
                                     nn.Linear(en2, en1),
                                     nn.Linear(en1, en0))

        self.x = torch.rand((64,50))
        u = torch.rand((69,5))
        self.U_set = []
        for step in range(2):
            self.U_set.append(u[step: 64+step])

    def recon(self, x):
        return self.decoder(self.encoder(x))

    def predict(self, x, u_set):
        z = self.encoder(x)
        z_next = z
        for u in u_set:
            Az = self.A(z_next)
            Bu = self.B(u)
            z_next = Az + Bu
        x_next = self.decoder(z_next)
        return x_next

    def forward(self, x, U_set):
        x_hat = self.recon(x)
        x_next = self.predict(x, U_set)

        return x_hat, x_next

nets = Nets(50,100,100,100,5)

with SummaryWriter("/home/sandymark/Tensorboard_logs/exp1") as w:
    w.add_graph(nets, [nets.x, nets.U_set])
