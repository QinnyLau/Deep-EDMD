import time

import Nets
from utils import *
from Config import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


if __name__ == '__main__':
    train_data = np.load("temp.npy")
    # veh_data = train_data.reshape((ngsim_data.shape[0], ngsim_data.shape[2] * ngsim_data.shape[3]))[:, :]
    norm_train, min_norm_train, max_norm_train = normalization(train_data)
    train_tensor = torch.tensor(norm_train, dtype=torch.float)
    print(train_tensor.shape)

    # test_data = np.load("data/" + EXPERIMENT_NAME + "_test.npy")
    # test_v = test_data.reshape((test_data.shape[0], test_data.shape[2] * test_data.shape[3]))[:, :]
    # norm_test, min_norm_test, max_norm_test = normalization(test_data)
    # test_veh_tensor = torch.tensor(norm_test, dtype=torch.float)

    ae = Nets.AutoEncoder(EN_IN, EN_HID1, EN_HID2, EN_OUT, AE_ACT_FUN)
    aux = Nets.Auxiliary(AUX_IN, AUX_HID1, AUX_HID2, AUX_HID3, AUX_OUT)
    # pred = Nets.Predict(60, 120, 120)

    mse = torch.nn.MSELoss()
    # ae_optimizer = torch.optim.Adam(ae.parameters(), LR)
    # aux_optimizer = torch.optim.Adam(aux.parameters(), LR)
    # pred_optimizer = torch.optim.Adam(itertools.chain(ae.parameters(), aux.parameters()), LR)
    optimizer = torch.optim.Adam(itertools.chain(ae.parameters(), aux.parameters()), LR)
    # t = torch.rand((100000, 20))

    rec_loss_log = []
    pred_loss_log = []
    aux_loss_log = []
    total_loss_log = []

    val_rec_loss_log = []
    val_pred_loss_log = []
    val_aux_loss_log = []
    val_total_loss_log = []

    for i in range(EPOCH):
        index = np.random.choice(train_tensor.shape[0] // SNAPSHOT_LENGTH - 1, BATCH_SIZE, False) * SNAPSHOT_LENGTH
        for j in tqdm(range(BATCH_SIZE)):
            # Reconstruct, optimize params of AE
            o = reconstruct(train_tensor[index[j]: index[j] + SNAPSHOT_LENGTH], ae)[1]
            rec_loss = mse(o, train_tensor[index[j]: index[j] + SNAPSHOT_LENGTH])

            # ae_optimizer.zero_grad()
            # rec_loss.backward()
            # ae_optimizer.step()

            # Predict, optimize params of Aux
            pred, code, code_pred = predict_return_all(train_tensor[index[j]: index[j] + SNAPSHOT_LENGTH], ae, aux)
            aux_loss = mse(code_pred, code[1:])

            # aux_optimizer.zero_grad()
            # aux_loss.backward()
            # aux_optimizer.step()

            # Predict, optimize params of AE and Aux
            # pred, _, _ = predict(train_tensor[index[j]: index[j] + SNAPSHOT_LENGTH])
            p_loss = mse(pred, train_tensor[index[j] + 1: index[j] + SNAPSHOT_LENGTH])
            #
            # pred_optimizer.zero_grad()
            # p_loss.backward()
            # pred_optimizer.step()
            loss = (R_CO * rec_loss + P_CO * p_loss) + A_CO * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if j % 16 == 0:
            #     print("rec_loss: ", rec_loss)
            #     print("p_loss: ", p_loss)

            # if i % 1 == 0:
            rec_loss_log.append(rec_loss.detach())
            pred_loss_log.append(p_loss.detach())
            aux_loss_log.append(aux_loss.detach())
            total_loss_log.append(loss.detach())

        if i % 5 == 0:
            print("----------------EPOCH %s---------------" % i)
            print("rec_loss: ", rec_loss)
            print("p_loss: ", p_loss)
            print("aux_loss: ", aux_loss)
            print("Total_loss: ", loss)
            print("=======================================")

            # index = np.random.choice(test_veh_tensor.shape[0], 1, False)[0]
            # o = reconstruct(test_veh_tensor[index: index + SNAPSHOT_LENGTH], ae)[1].detach()
            # rec_loss = mse(o, test_veh_tensor[index: index + SNAPSHOT_LENGTH])
            #
            # pred, code, code_pred = predict(test_veh_tensor[index: index + SNAPSHOT_LENGTH], ae, aux)
            # aux_loss = mse(code_pred, code[1:])
            # p_loss = mse(pred, test_veh_tensor[index + 1: index + SNAPSHOT_LENGTH])
            #
            # loss = (R_CO * rec_loss + P_CO * p_loss) + A_CO * aux_loss

            # print(">>>>>>>>>>>>>>>>>EVAL>>>>>>>>>>>>>>>>>>")
            # print("rec_loss: ", rec_loss)
            # print("p_loss: ", p_loss)
            # print("aux_loss: ", aux_loss)
            # print("Total_loss: ", loss)
            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #
            # val_rec_loss_log.append(rec_loss)
            # val_pred_loss_log.append(p_loss)
            # val_aux_loss_log.append(aux_loss)
            # val_total_loss_log.append(loss)

        # p1, _, _ = predict(torch.unsqueeze(train_tensor[100], 0))
        # p1_loss = mse(p1.detach(), torch.unsqueeze(train_tensor[101], 0))
        #
        # plt.figure(1)
        # plt.scatter(np.linspace(0, len(train_tensor[100]) - 1, len(train_tensor[100])),
        #             denormalization(train_tensor[100], min_norm_test, max_norm_test), color='r')
        # plt.scatter(np.linspace(0, len(train_tensor[100]) - 1, len(train_tensor[100])),
        #             denormalization(reconstruct(torch.unsqueeze(train_tensor[100], 0))[1].detach(), min_norm_test,
        #                             max_norm_test), color='g')
        #
        # plt.figure(2)
        # plt.scatter(np.linspace(0, len(train_tensor[101]) - 1, len(train_tensor[101])),
        #             denormalization(train_tensor[101], min_norm_test, max_norm_test), color='r')
        # plt.scatter(np.linspace(0, len(train_tensor[101]) - 1, len(train_tensor[101])),
        #             denormalization(p1.detach(), min_norm_test, max_norm_test), color='g')
        # plt.show()

    try:
        np.savetxt("log/recon_loss.csv", rec_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log/pred_loss.csv", pred_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log/aux_loss.csv", aux_loss_log, fmt='%.10f', delimiter=',')
        np.savetxt("log/total_loss.csv", total_loss_log, fmt='%f', delimiter=',')

        np.savetxt("log/val_recon_loss.csv", val_rec_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log/val_pred_loss.csv", val_pred_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log/val_aux_loss.csv", val_aux_loss_log, fmt='%.10f', delimiter=',')
        np.savetxt("log/val_total_loss.csv", val_total_loss_log, fmt='%f', delimiter=',')
    except:
        print("Log save FAILED!!")

    try:
        time_str = time.strftime('%m%d-%H-%M', time.localtime(time.time()))
        torch.save(ae, "saved/" + STATE_VALUE + "_" + time_str + "AE.pth")
        torch.save(aux, "saved/" + STATE_VALUE + "_" + time_str + "AUX.pth")
    except:
        print("Model save FAILED!!")

    # test start
    # test_rec_loss = []
    # test_pred_loss = []
    # test_aux_loss = []
    # test_total_loss = []

    # for i in tqdm(range(TEST_BATCH)):
    #     index = np.random.choice(test_veh_tensor.shape[0] // SNAPSHOT_LENGTH - 1, 1, False)[0] * SNAPSHOT_LENGTH
    #     o = reconstruct(train_tensor[index: index + SNAPSHOT_LENGTH], ae)[1]
    #     rec_loss = mse(o, train_tensor[index: index + SNAPSHOT_LENGTH])
    #
    #     pred, code, code_pred = predict(train_tensor[index: index + SNAPSHOT_LENGTH], ae, aux)
    #     aux_loss = mse(code_pred, code[1:])
    #
    #     pred_loss = mse(pred, train_tensor[index + 1: index + SNAPSHOT_LENGTH])
    #
    #     test_rec_loss.append(rec_loss.data)
    #     test_pred_loss.append(pred_loss.data)
    #     test_aux_loss.append(aux_loss.data)
    #     test_total_loss.append((R_CO * rec_loss.data + P_CO * pred_loss.data) + A_CO * aux_loss.data)
    #
    # print("\n+++++++++++++++++TEST RESULT++++++++++++++++++")
    # print("AVG_REC_LOSS: ", np.average(test_rec_loss))
    # print("AVG_PRED_LOSS: ", np.average(test_pred_loss))
    # print("AVG_AUX_LOSS: ", np.average(test_aux_loss))
    # print("AVG_TOTAL_LOSS: ", np.average(test_total_loss))
    # print("+++++++++++++++++++++++++++++++++++++++++++++")
    # print("STD_REC_LOSS: ", np.std(test_rec_loss))
    # print("STD_PRED_LOSS: ", np.std(test_pred_loss))
    # print("STD_AUX_LOSS: ", np.std(test_aux_loss))
    # print("STD_TOTAL_LOSS: ", np.std(test_total_loss))
    # print("+++++++++++++++++++++++++++++++++++++++++++++")
    # print("MAX_REC_LOSS: ", np.max(test_rec_loss))
    # print("MAX_PRED_LOSS: ", np.max(test_pred_loss))
    # print("MAX_AUX_LOSS: ", np.max(test_aux_loss))
    # print("MAX_TOTAL_LOSS: ", np.max(test_total_loss))
    # print("+++++++++++++++++++++++++++++++++++++++++++++\n")
    #
    # # index = np.random.choice(test_veh_tensor.shape[0] - 1, 64, False)
    # o = reconstruct(torch.unsqueeze(test_veh_tensor[260],0), ae)[1]
    # rec_loss = mse(o, torch.unsqueeze(test_veh_tensor[260],0))
    # #
    # p1, c1, cp1 = predict(train_tensor[260: 262], ae, aux)
    # p1_loss = mse(p1, torch.unsqueeze(train_tensor[261],0))
    # aux1_loss = mse(cp1, c1[1:])
    # #
    # # p2 = predict(p1.detach())
    # # p2_loss = mse(p2, test_veh_tensor[100 + 2])
    # #
    # # p3 = predict(p2.detach())
    # # p3_loss = mse(p3, test_veh_tensor[100 + 3])
    # #
    # # p4 = predict(p3.detach())
    # # p4_loss = mse(p4, test_veh_tensor[100 + 4])
    # #
    # # p5 = predict(p4.detach())
    # # p5_loss = mse(p5, test_veh_tensor[100 + 5])
    # #
    # print("test rec_loss: ", rec_loss)
    # print("test pred_loss1: ", p1_loss)
    # print("test aux_loss: ", aux1_loss)
    # print("test total loss: ", (R_CO * rec_loss + P_CO * p_loss) + A_CO * aux_loss)
    # # print("test pred_loss2: ", p2_loss)
    # # print("test pred_loss3: ", p3_loss)
    # # print("test pred_loss4: ", p4_loss)
    # # print("test pred_loss5: ", p5_loss)
    # print("==================")
    # print(denormalization(train_tensor[260], min_norm_test, max_norm_test))
    # print("encoder out:")
    # print(denormalization(reconstruct(torch.unsqueeze(train_tensor[260],0), ae)[0].detach(), min_norm_test, max_norm_test))
    # print("decoder out:")
    # print(denormalization(reconstruct(torch.unsqueeze(train_tensor[260],0), ae)[1].detach(), min_norm_test, max_norm_test))
    # print("------------------")
    # print("t[101]: ", denormalization(train_tensor[261], min_norm_test, max_norm_test))
    # print(denormalization(predict(torch.unsqueeze(train_tensor[260],0), ae, aux)[0].detach(), min_norm_test, max_norm_test))
    #
    # plt.figure(1)
    # plt.scatter(np.linspace(0, len(train_tensor[260])-1, len(train_tensor[260])),
    #             denormalization(train_tensor[260], min_norm_test, max_norm_test), color='r')
    # plt.scatter(np.linspace(0, len(train_tensor[260])-1, len(train_tensor[260])),
    #             denormalization(reconstruct(torch.unsqueeze(train_tensor[260],0), ae)[1].detach(), min_norm_test, max_norm_test), color='g')
    #
    # # testTensor = torch.tensor([1.,1.,1.,1.,0.1,1.,1.,0.02,0.8,0.9,0.8,0.8,0.8,0.8,1.,0.,0.,0.,1.,1.])
    # # testOut = reconstruct(torch.unsqueeze(testTensor, 0))[1]
    # # plt.figure(2)
    # # plt.scatter(np.linspace(0, len(testTensor)-1, len(testTensor)), testTensor, color='r')
    # # plt.scatter(np.linspace(0, len(testTensor)-1, len(testTensor)), testOut.detach()[0], color='g')
    #
    # plt.figure(2)
    # plt.scatter(np.linspace(0, len(train_tensor[261])-1, len(train_tensor[261])),
    #             denormalization(train_tensor[261], min_norm_test, max_norm_test), color='r')
    # plt.scatter(np.linspace(0, len(train_tensor[261])-1, len(train_tensor[261])),
    #             denormalization(p1.detach(), min_norm_test, max_norm_test), color='g')
    # #
    # # plt.figure(3)
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[102])-1, len(test_veh_tensor[102])),
    # #             denormalization(test_veh_tensor[102], min_norm_test_v, max_norm_test_v), color='r')
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[102])-1, len(test_veh_tensor[102])),
    # #             denormalization(p2.detach(), min_norm_test_v, max_norm_test_v), color='g')
    # #
    # # plt.figure(4)
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[103])-1, len(test_veh_tensor[103])),
    # #             denormalization(test_veh_tensor[103], min_norm_test_v, max_norm_test_v), color='r')
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[103])-1, len(test_veh_tensor[103])),
    # #             denormalization(p3.detach(), min_norm_test_v, max_norm_test_v), color='g')
    # #
    # # plt.figure(5)
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[104])-1, len(test_veh_tensor[104])),
    # #             denormalization(test_veh_tensor[104], min_norm_test_v, max_norm_test_v), color='r')
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[104])-1, len(test_veh_tensor[104])),
    # #             denormalization(p4.detach(), min_norm_test_v, max_norm_test_v), color='g')
    # #
    # # plt.figure(6)
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[105])-1, len(test_veh_tensor[105])),
    # #             denormalization(test_veh_tensor[105], min_norm_test_v, max_norm_test_v), color='r')
    # # plt.scatter(np.linspace(0, len(test_veh_tensor[105])-1, len(test_veh_tensor[105])),
    # #             denormalization(p5.detach(), min_norm_test_v, max_norm_test_v), color='g')
    #
    # plt.show()
