#import Nets
from Nets import *
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    #ngsim_data = np.load("../NGSIM/data/us101trj/train_labels_lookback0_1-5_noscale.npy")
    a = np.loadtxt('I10_East_Week_Velocity_Data.txt')
    #print(a)
    #b=a[:,:]
    np.save("data/NGSIM/I10_East_Week_Velocity_Data_test.npy", a)
    #ngsim_data = np.load("NGSIM_US80_5pm_Velocity_Data.npy", allow_pickle=True)
    ngsim_data=a
    print(ngsim_data.shape)

    v_data = ngsim_data.reshape((ngsim_data.shape[0], ngsim_data.shape[1]))[:, :]
    print(v_data)
    norm_v, min_norm_v, max_norm_v = normalization(v_data)
    v_tensor = torch.tensor(norm_v, dtype=torch.float)
    print(v_tensor)

   # test_data = np.load("../NGSIM/data/us101trj/eval_labels_lookback0_1-5_noscale.npy")
    test_data = np.loadtxt("NGSIM_US80_5pm_Velocity_Data.txt")
    #test_v = test_data.reshape((test_data.shape[0], test_data.shape[2] * test_data.shape[3]))[:, :]
    test_v = test_data.reshape((test_data.shape[0], test_data.shape[1]))[:, :]
    norm_test_v, min_norm_test_v, max_norm_test_v = normalization(test_v)
    test_v_tensor = torch.tensor(norm_test_v, dtype=torch.float)

    ae = Nets(30, 180 ,180, 30, 1)
    #ae = Nets.recon(30, 120, 120, 60, 'relu')
    # aux = Nets.Auxiliary(2, 10, 10, 10, 2)
    #pred = Nets.Predict(60, 120, 120)
    pred = Nets(30, 180, 180,30,1)

    def reconstruct(data):
        return ae(data)


    def predict(data):
        code = ae.encode(data).detach()
        pred_code = pred(code)
        predict = ae.decode(pred_code)

        return predict


    mse = torch.nn.MSELoss()
    ae_optimizer = torch.optim.Adam(ae.parameters(), 1e-3)
    pred_optimizer = torch.optim.Adam(pred.parameters(), 1e-3)
    # t = torch.rand((100000, 20))

    for i in tqdm(range(10000)):
        index = np.random.choice(v_tensor.shape[0] - 1, 30, False)
        print(index)
        o = reconstruct(v_tensor[index])[1]
        rec_loss = mse(o, v_tensor[index])

        ae_optimizer.zero_grad()
        rec_loss.backward()
        ae_optimizer.step()

        p = predict(v_tensor[index])
        p_loss = mse(p, v_tensor[index + 1])

        pred_optimizer.zero_grad()
        p_loss.backward()
        pred_optimizer.step()

        if i % 500 == 0:
            print("rec_loss: ", rec_loss)
            print("p_loss: ", p_loss)

    # index = np.random.choice(test_v_tensor.shape[0] - 1, 64, False)
    o = reconstruct(test_v_tensor[100])[1]
    rec_loss = mse(o, test_v_tensor[100])

    p1 = predict(test_v_tensor[100])
    p1_loss = mse(p1, test_v_tensor[100 + 1])

    p2 = predict(p1.detach())
    p2_loss = mse(p2, test_v_tensor[100 + 2])

    p3 = predict(p2.detach())
    p3_loss = mse(p3, test_v_tensor[100 + 3])

    p4 = predict(p3.detach())
    p4_loss = mse(p4, test_v_tensor[100 + 4])

    p5 = predict(p4.detach())
    p5_loss = mse(p5, test_v_tensor[100 + 5])

    print("test rec_loss: ", rec_loss)
    print("test pred_loss1: ", p1_loss)
    print("test pred_loss2: ", p2_loss)
    print("test pred_loss3: ", p3_loss)
    print("test pred_loss4: ", p4_loss)
    print("test pred_loss5: ", p5_loss)
    print("==================")
    print(denormalization(test_v_tensor[100], min_norm_test_v, max_norm_test_v))
    print("encoder out:")
    print(denormalization(reconstruct(test_v_tensor[100])[0].detach(), min_norm_test_v, max_norm_test_v))
    print("decoder out:")
    print(denormalization(reconstruct(test_v_tensor[100])[1].detach(), min_norm_test_v, max_norm_test_v))
    print("------------------")
    print("t[101]: ", denormalization(test_v_tensor[101], min_norm_test_v, max_norm_test_v))
    print(denormalization(predict(test_v_tensor[100]).detach(), min_norm_test_v, max_norm_test_v))

    plt.figure(1)
    plt.scatter(np.linspace(0, len(test_v_tensor[100])-1, len(test_v_tensor[100])),
                denormalization(test_v_tensor[100], min_norm_test_v, max_norm_test_v), color='r')
    plt.scatter(np.linspace(0, len(test_v_tensor[100])-1, len(test_v_tensor[100])),
                denormalization(reconstruct(test_v_tensor[100])[1].detach(), min_norm_test_v, max_norm_test_v), color='g')

    # testTensor = torch.tensor([1.,1.,1.,1.,0.1,1.,1.,0.02,0.8,0.9,0.8,0.8,0.8,0.8,1.,0.,0.,0.,1.,1.])
    # testOut = reconstruct(torch.unsqueeze(testTensor, 0))[1]
    # plt.figure(2)
    # plt.scatter(np.linspace(0, len(testTensor)-1, len(testTensor)), testTensor, color='r')
    # plt.scatter(np.linspace(0, len(testTensor)-1, len(testTensor)), testOut.detach()[0], color='g')

    plt.figure(2)
    plt.scatter(np.linspace(0, len(test_v_tensor[101])-1, len(test_v_tensor[101])),
                denormalization(test_v_tensor[101], min_norm_test_v, max_norm_test_v), color='r')
    plt.scatter(np.linspace(0, len(test_v_tensor[101])-1, len(test_v_tensor[101])),
                denormalization(p1.detach(), min_norm_test_v, max_norm_test_v), color='g')

    plt.figure(3)
    plt.scatter(np.linspace(0, len(test_v_tensor[102])-1, len(test_v_tensor[102])),
                denormalization(test_v_tensor[102], min_norm_test_v, max_norm_test_v), color='r')
    plt.scatter(np.linspace(0, len(test_v_tensor[102])-1, len(test_v_tensor[102])),
                denormalization(p2.detach(), min_norm_test_v, max_norm_test_v), color='g')

    plt.figure(4)
    plt.scatter(np.linspace(0, len(test_v_tensor[103])-1, len(test_v_tensor[103])),
                denormalization(test_v_tensor[103], min_norm_test_v, max_norm_test_v), color='r')
    plt.scatter(np.linspace(0, len(test_v_tensor[103])-1, len(test_v_tensor[103])),
                denormalization(p3.detach(), min_norm_test_v, max_norm_test_v), color='g')

    plt.figure(5)
    plt.scatter(np.linspace(0, len(test_v_tensor[104])-1, len(test_v_tensor[104])),
                denormalization(test_v_tensor[104], min_norm_test_v, max_norm_test_v), color='r')
    plt.scatter(np.linspace(0, len(test_v_tensor[104])-1, len(test_v_tensor[104])),
                denormalization(p4.detach(), min_norm_test_v, max_norm_test_v), color='g')

    plt.figure(6)
    plt.scatter(np.linspace(0, len(test_v_tensor[105])-1, len(test_v_tensor[105])),
                denormalization(test_v_tensor[105], min_norm_test_v, max_norm_test_v), color='r')
    plt.scatter(np.linspace(0, len(test_v_tensor[105])-1, len(test_v_tensor[105])),
                denormalization(p5.detach(), min_norm_test_v, max_norm_test_v), color='g')

    plt.show()
