import torch.nn as nn


class Nets(nn.Module):
    def __init__(self, en0, en1, en2, en3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(en0, en1, bias=False),
                                     nn.Tanhshrink(),
                                     nn.Linear(en1, en2, bias=False),
                                     nn.Tanhshrink(),
                                     nn.Linear(en2, en3, bias=False),)
        self.A = nn.Linear(en3, en3, bias=False)
       # self.B = nn.Linear(b0, en3, bias=False)
        # self.A = nn.Sequential(nn.Linear(en3, en3, bias=False),
        #                        nn.ReLU(),
        #                        nn.Linear(en3, en3, bias=False))
        # self.B = nn.Sequential(nn.Linear(b0, en3, bias=False),
        #                        nn.ReLU(),
        #                        nn.Linear(en3, en3, bias=False))
        self.decoder = nn.Sequential(nn.Linear(en3, en0, bias=False),)
                                     # nn.ReLU(),
                                     # nn.Linear(en2, en1, bias=False),
                                     # nn.PReLU(),
                                     # nn.Linear(en1, en0, bias=False))

    # def reconstruct(self, x):
    #     z = self.encoder(x)
    #     x_hat = self.decoder(z)
    #
    #     return x_hat
    #
    # def predict(self, x, u):
    #     z = self.encoder(x)
    #     Az = self.A(z)
    #     Bu = self.B(u)
    #     z_next = Az + Bu
    #     x_next = self.decoder(z_next)
    #
    #     return x_next
    def recon(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def predict(self, x):
        z = self.encoder(x).detach()
        z_next = z
        Az = self.A(z_next)
        z_next = Az
        # for u in u_set:
        #     Az = self.A(z_next)
        #     Bu = self.B(u)
        #     z_next = Az + Bu
        x_next = self.decoder(z_next)
        return x_next, z_next



