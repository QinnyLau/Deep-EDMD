#import Nets
from Nets import *
from util_new import *
from LoadConfig import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import time

if __name__ == '__main__':
    #writer = SummaryWriter(comment="DeepKoopman")

    train_tensor, dim_train_state = load_data_with_ctrl("train", STATE_SEQS)
    test_tensor, dim_test_state = load_data_with_ctrl("test", STATE_SEQS)
    nets = Nets(EN_IN, EN_HID1, EN_HID2, EN_OUT)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nets.parameters(), LR)

    rec_loss_log = []
    pred_loss_log = []
    lin_loss_log = []
    total_loss_log = []

    val_rec_loss_log = []
    val_pred_loss_log = []
    val_lin_loss_log = []
    val_total_loss_log = []

    rec = []
    train_index_list = np.array([], dtype=int)
    for j in range(NUM_TRAIN_TRAJ):
        train_index_list = np.concatenate((train_index_list,
                                           np.linspace(j * SNAPSHOT_TRAIN_LEN,
                                                       (j + 1) * SNAPSHOT_TRAIN_LEN - PRED_LENGTH - 1,
                                                       SNAPSHOT_TRAIN_LEN - PRED_LENGTH, dtype=int)))
    test_index_list = np.array([], dtype=int)
    for j in range(NUM_TEST_TRAJ):
        test_index_list = np.concatenate((test_index_list,
                                          np.linspace(j * SNAPSHOT_TEST_LEN,
                                                      (j + 1) * SNAPSHOT_TEST_LEN - PRED_LENGTH - 1,
                                                      SNAPSHOT_TEST_LEN - PRED_LENGTH, dtype=int)))

    for i in tqdm(range(1, EPOCH + 1)):
        index = np.random.choice(train_index_list, BATCH_SIZE, False)
        X = train_tensor[index, :DIM_STATE]
        #U_set = []
        #for step in range(PRED_LENGTH):
            #U_set.append(train_tensor[index + step, -DIM_CTRL:])

        X_hat = nets.recon(X)
        rec_loss = mse(X, X_hat)

        # optimizer.zero_grad()
        # rec_loss.backward()
        # optimizer.step()

        X_next, Z_next = nets.predict(X)
        p_loss = mse(train_tensor[index + PRED_LENGTH, :DIM_STATE], X_next) / PRED_LENGTH
        lin_loss = mse(nets.encoder(train_tensor[index + PRED_LENGTH, :DIM_STATE]).detach(), Z_next) / PRED_LENGTH
        loss = rec_loss + P_CO * p_loss + A_CO * lin_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer.add_scalar("recon_loss", rec_loss.item(), global_step=i)
        # writer.add_scalar("predict_loss", p_loss.item(), global_step=i)
        # writer.add_scalar("linear_loss", lin_loss.item(), global_step=i)
        # writer.add_scalar("total_loss", loss.item(), global_step=i)


        rec_loss_log.append(rec_loss.detach())
        pred_loss_log.append(p_loss.detach())
        lin_loss_log.append(lin_loss.detach())
        total_loss_log.append(loss.detach())

        if i % 20 == 0:
            print("----------------EPOCH %s---------------\n"
                  "rec_loss: %.5f\n"
                  "p_loss: %.5f\n"
                  "lin_loss: %.5f\n"
                  "Total_loss: %.5f\n"
                  "======================================="
                  % (i, rec_loss.detach(), p_loss.detach(), lin_loss.detach(), loss.detach()))

            index = np.random.choice(test_index_list, BATCH_SIZE, False)
            # U_set = []
            # for step in range(PRED_LENGTH):
            #     U_set.append(test_tensor[index + step, -DIM_CTRL:])
            o = nets.recon(test_tensor[index, :DIM_STATE]).detach()
            print(o.shape)
            rec_loss = mse(o, test_tensor[index, :DIM_STATE]).detach()

            X_next, Z_next = nets.predict(test_tensor[index, :DIM_STATE])
            print(X_next.shape)
            p_loss = mse(test_tensor[index + PRED_LENGTH, :DIM_STATE], X_next).detach() / PRED_LENGTH
            lin_loss = mse(nets.encoder(test_tensor[index + PRED_LENGTH, :DIM_STATE]).detach(), Z_next).detach() / PRED_LENGTH
            loss = R_CO * rec_loss + P_CO * p_loss + A_CO * lin_loss

            print(">>>>>>>>>>>>>>>>>EVAL>>>>>>>>>>>>>>>>>>\n"
                  "rec_loss: %.5f\n"
                  "p_loss: %.5f\n"
                  "lin_loss: %.5f\n"
                  "Total_loss: %.5f\n"
                  ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
                  % (rec_loss, p_loss, lin_loss, loss))

            val_rec_loss_log.append(rec_loss)
            val_pred_loss_log.append(p_loss)
            val_lin_loss_log.append(lin_loss)
            val_total_loss_log.append(loss)

    try:
        np.savetxt("log1/recon_loss.csv", rec_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log1/pred_loss.csv", pred_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log1/lin_loss.csv", lin_loss_log, fmt='%.10f', delimiter=',')
        np.savetxt("log1/total_loss.csv", total_loss_log, fmt='%f', delimiter=',')
        #np.savetxt("log/rec.csv", rec, fmt='%f', delimiter=',')

        np.savetxt("log1/val_recon_loss.csv", val_rec_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log1/val_pred_loss.csv", val_pred_loss_log, fmt='%f', delimiter=',')
        np.savetxt("log1/val_aux_loss.csv", val_lin_loss_log, fmt='%.10f', delimiter=',')
        np.savetxt("log1/val_total_loss.csv", val_total_loss_log, fmt='%f', delimiter=',')
    except:
        print("Log save FAILED!!")

    try:
        time_str = time.strftime('%m%d-%H-%M', time.localtime(time.time()))
        torch.save(nets, NET_SAVE_DIR + "_" + time_str + ".pth")
    except:
        print("Model save FAILED!!")

    # test start
    test_rec_loss = []
    test_pred_loss = []
    test_lin_loss = []
    test_total_loss = []

    for i in tqdm(range(TEST_BATCH)):
        index = np.random.choice(test_index_list, BATCH_SIZE, False)
        # U_set = []
        # for step in range(PRED_LENGTH):
        #     U_set.append(test_tensor[index + step, -DIM_CTRL:])
        o = nets.recon(test_tensor[index, :DIM_STATE]).detach()
        rec_loss = mse(o, test_tensor[index, :DIM_STATE]).detach()

        X_pred, Z_pred = nets.predict(test_tensor[index, :DIM_STATE])
        p_loss = mse(X_pred, test_tensor[index + PRED_LENGTH, :DIM_STATE]).detach() / PRED_LENGTH
        l_loss = mse(Z_pred, nets.encoder(test_tensor[index + PRED_LENGTH, :DIM_STATE])).detach() / PRED_LENGTH

        test_rec_loss.append(rec_loss)
        test_pred_loss.append(p_loss)
        test_lin_loss.append(l_loss)
        test_total_loss.append(R_CO * rec_loss + P_CO * p_loss + A_CO * l_loss)

    print("\n+++++++++++++++++TEST RESULT++++++++++++++++++\n"
          "AVG_REC_LOSS: %.5f\n"
          "AVG_PRED_LOSS: %.5f\n"
          "AVG_LINEAR_LOSS: %.5f\n"
          "AVG_TOTAL_LOSS: %.5f\n"
          "+++++++++++++++++++++++++++++++++++++++++++++\n"
          "STD_REC_LOSS: %.5f\n"
          "STD_PRED_LOSS: %.5f\n"
          "STD_LINEAR_LOSS: %.5f\n"
          "STD_TOTAL_LOSS: %.5f\n"
          "+++++++++++++++++++++++++++++++++++++++++++++\n"
          "MAX_REC_LOSS: %.5f\n"
          "MAX_PRED_LOSS: %.5f\n"
          "MAX_LINEAR_LOSS: %.5f\n"
          "MAX_TOTAL_LOSS: %.5f\n"
          "+++++++++++++++++++++++++++++++++++++++++++++\n"
          % (np.mean(test_rec_loss), np.mean(test_pred_loss), np.mean(test_lin_loss), np.mean(test_total_loss),
             np.std(test_rec_loss), np.std(test_pred_loss), np.std(test_lin_loss), np.std(test_total_loss),
             np.max(test_rec_loss), np.max(test_pred_loss), np.max(test_lin_loss), np.max(test_total_loss)))

    try:
        while True:

            index = int(input())
            #index = 540

            o = nets.recon(test_tensor[index, :DIM_STATE]).detach()

            p = torch.tensor([])

            for i in range(20):
                # U_set = []
                # for step in range(i+1):
                #     U_set.append(test_tensor[index + step, -DIM_CTRL:].unsqueeze(0))
                _p = nets.predict(test_tensor[index, :DIM_STATE].unsqueeze(0))[0].detach()
                print(mse(_p, test_tensor[index + i, :DIM_STATE].unsqueeze(0)))
                p = torch.cat((p, _p), dim=0)
            print(p.shape)
            std = test_tensor[index: index+20, :DIM_STATE].numpy()
            plt.figure()
            plt.plot(np.linspace(1,20,20,dtype=int),std[:, 0], 'r^--')
            plt.plot(np.linspace(1,20,20,dtype=int),p.numpy()[:, 0], 'g^--')
            plt.xlim(0,20)
            plt.xticks(range(0,21,2))
            plt.xlabel("predict step")
            plt.ylabel("velocity")
            plt.legend(['origin', 'predict'])
            plt.grid()
            plt.figure()
            plt.plot(np.linspace(1,20,20,dtype=int),std[:, 25], 'r^--')
            plt.plot(np.linspace(1,20,20,dtype=int),p.numpy()[:, 25], 'g^--')
            plt.xlim(0,20)
            plt.xticks(range(0,21,2))
            plt.xlabel("predict step")
            plt.ylabel("spacing")
            plt.legend(['origin', 'predict'])
            plt.grid()
            plt.show()

    except:
        pass

    while True:
        idx = int(input())
        shown_index = np.array([idx])
        o = nets.recon(test_tensor[shown_index, :DIM_STATE]).detach()
        # U_set = []
        # for step in range(20):
        #     U_set.append(test_tensor[shown_index + step, -DIM_CTRL:])
        p = nets.predict(test_tensor[shown_index, :DIM_STATE])[0].detach()

        print("==================")
        print(test_tensor[shown_index].to("cpu"))
        print("decoder out:")
        print(o.detach().to("cpu"))
        print("------------------")

        plt.figure(1)
        plt.plot(
            np.linspace(0, len(test_tensor[shown_index[0]][:DIM_STATE]) - 1, len(test_tensor[shown_index[0]][:DIM_STATE])),
            test_tensor[shown_index, :DIM_STATE].to("cpu")[0], 'r^--')
        plt.plot(
            np.linspace(0, len(test_tensor[shown_index[0]][:DIM_STATE]) - 1, len(test_tensor[shown_index[0]][:DIM_STATE])),
            o.detach().to("cpu")[0], 'g^--')
        plt.legend(["origin data", "predict data"])
        plt.xlabel("state dimension")
        plt.ylabel("value")

        # plt.figure(2)
        plt.plot(np.linspace(0, len(test_tensor[shown_index[0] + 20][:DIM_STATE]) - 1,
                             len(test_tensor[shown_index[0] + 20][:DIM_STATE])),
                 test_tensor[shown_index + 20, :DIM_STATE].to("cpu")[0], 'b^--')
        plt.plot(np.linspace(0, len(test_tensor[shown_index[0] + 20][:DIM_STATE]) - 1,
                             len(test_tensor[shown_index[0] + 20][:DIM_STATE])),
                 p.detach().to("cpu")[0])
        plt.legend(["origin data", "predict data"])
        plt.xlabel("state dimension")
        plt.ylabel("value")
        plt.show()

    print("A:", nets.state_dict()["A.weight"])
    print("B:", nets.state_dict()["B.weight"])