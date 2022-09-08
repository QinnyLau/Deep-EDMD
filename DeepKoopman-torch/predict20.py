import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *


if __name__ == '__main__':
    index = 16415
    train_tensor = load_data_with_ctrl("train", [0], [0])[0]
    nets = torch.load("./saved/IDM_distMacro_0228-00-07.pth")
    ylabel = ["v (m/s)", "d (m)", "d_v (m/s)", "d_rho"]

    predm = torch.tensor([])
    for i in range(20):
        pred = nets.predict(train_tensor[index+i: index+i+1, :4], train_tensor[index+i: index+i+1, 4:5])[0].detach()
        predm = torch.cat((predm, pred), 0)

    for i in range(4):
        plt.figure(figsize=(6,3))
        plt.grid()
        plt.plot(train_tensor[index+1:index+21, i:i+1].numpy(), '^--')
        plt.plot(predm[:, i:i+1].numpy(), '^--')
        plt.xlabel("predict step")
        plt.ylabel(ylabel[i])
        plt.xlim(0,20)
        plt.legend(["real value", "predictive value"])
        plt.tight_layout()

    plt.show()
