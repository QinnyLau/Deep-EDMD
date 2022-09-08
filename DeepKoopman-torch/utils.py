import numpy as np
import torch
from LoadConfig import *


def load_data_with_ctrl(data_type, state_sqs: list = None, ctrl_sqs: list = None):
    if data_type not in ["train", "test"]:
        raise Exception("data type cannot be recognized. (`train' or `test')")
    data_type = '_' + data_type + ".npy"

    # load states value
    states = None

    for s in STATE_VALUE:
        temp_state = np.load("data/" + s + data_type)
        if s == 'macro/IDM_dcav_traj':
            states = np.concatenate((states, temp_state), axis=1) if states is not None else temp_state
        else:
            if state_sqs is not None:
                states = np.concatenate((states, temp_state[:, state_sqs]), axis=1) \
                    if states is not None else temp_state[:, state_sqs]
            else:
                states = np.concatenate((states, temp_state), axis=1) if states is not None else temp_state

    # load control value
    controls = np.load("data/" + CONTROL_VALUE + data_type)
    if ctrl_sqs:
        controls = controls[:, ctrl_sqs]

    # aggregate
    data = np.hstack((states, controls))
    data_tensor = torch.tensor(data, dtype=torch.float).to(DEVICE)

    return data_tensor, states.shape[1]


def normalization(data):
    if isinstance(data, torch.Tensor):
        try:
            data = data.numpy()
        except RuntimeError:
            data = data.detach.numpy()

    min = 0
    data -= min
    max = 1
    data /= max

    # return data * 2 - 1, min, max
    return data, min, max


# def normalization(data, min, max):
#     if isinstance(data, torch.Tensor):
#         try:
#             data = data.numpy()
#         except RuntimeError:
#             data = data.detach.numpy()
#
#     data -= min
#     data /= max
#
#     # return data * 2 - 1, min, max
#     return data, min, max


def denormalization(data, min, max):
    if isinstance(data, torch.Tensor):
        try:
            data = data.numpy()
        except RuntimeError:
            data = data.detach.numpy()

    return (data + 1) / 2 * max + min


def denormalization_with_ctrl(data, dim_state, min_state, max_state, min_ctrl, max_ctrl):
    if isinstance(data, torch.Tensor):
        try:
            data = data.numpy()
        except RuntimeError:
            data = data.detach.numpy()

    state_data = data[:, : dim_state]
    ctrl_data = data[:, dim_state:]
    state_data = (state_data + 1) / 2 * max_state + min_state
    ctrl_data = (ctrl_data + 1) / 2 * max_ctrl + min_ctrl
    # state_data = state_data * max_state + min_state
    # ctrl_data = ctrl_data * max_ctrl + min_ctrl

    return np.hstack((state_data, ctrl_data))


def get_radius(code: torch.Tensor):
    subMatrix = torch.pow(code, 2).unsqueeze(1)
    subMatrix = subMatrix.reshape((subMatrix.shape[0], subMatrix.shape[2] // 2, 2)).permute(1, 0, 2)

    return torch.sum(subMatrix, -1).T


# def get_K(eigenvalue: torch.Tensor):
#     subMatirx = torch.split(eigenvalue, 2, dim=1)
#     K = torch.zeros((eigenvalue.shape[0], len(subMatirx) * 2, len(subMatirx) * 2))
#     for i in range(K.shape[0]):
#         for j in range(len(subMatirx)):
#             element_1 = torch.exp(subMatirx[j][0][0] * deltaT) * torch.cos(subMatirx[j][0][1] * deltaT)
#             element_2 = torch.exp(subMatirx[j][0][0] * deltaT) * torch.sin(subMatirx[j][0][1] * deltaT)
#             K[i][j * 2][j * 2] = element_1
#             K[i][j * 2][j * 2 + 1] = - element_2
#             K[i][j * 2 + 1][j * 2] = element_2
#             K[i][j * 2 + 1][j * 2 + 1] = element_1
#
#     return K

# def get_K(eigenvalue: torch.Tensor):
#     subMatrix = torch.unsqueeze(eigenvalue, 1).reshape((eigenvalue.shape[0], eigenvalue.shape[1] // 2, 2))
#     real_part = torch.exp(subMatrix[:, :, 0] * deltaT)
#     element1 = real_part * torch.cos(subMatrix[:, :, 1] * deltaT)
#     element2 = real_part * torch.sin(subMatrix[:, :, 1] * deltaT)
#     K = torch.zeros((eigenvalue.shape[0], subMatrix.shape[1] * 2, subMatrix.shape[1] * 2))
#     for i in range(K.shape[0]):
#         for j in range(subMatrix.shape[1]):
#             K[i][j * 2][j * 2] = element1[i][j]
#             K[i][j * 2][j * 2 + 1] = - element2[i][j]
#             K[i][j * 2 + 1][j * 2] = element2[i][j]
#             K[i][j * 2 + 1][j * 2 + 1] = element1[i][j]
#
#     return K

def get_K(eigenvalue: torch.Tensor):
    subMatrix = torch.unsqueeze(eigenvalue, 1).reshape((eigenvalue.shape[0], eigenvalue.shape[1] // 2, 2))
    real_part = torch.exp(subMatrix[:, :, 0] * deltaT)
    element1 = real_part * torch.cos(subMatrix[:, :, 1] * deltaT)
    element2 = real_part * torch.sin(subMatrix[:, :, 1] * deltaT)
    K = torch.zeros((eigenvalue.shape[0], DIM_STATE * 2, DIM_STATE * 2))
    for i in range(K.shape[0]):
        for j in range(DIM_STATE):
            K[i][j * 2][j * 2] = element1[i][j]
            K[i][j * 2][j * 2 + 1] = - element2[i][j]
            K[i][j * 2 + 1][j * 2] = element2[i][j]
            K[i][j * 2 + 1][j * 2 + 1] = element1[i][j]

    return K


# def get_K(eigenvalue: torch.Tensor):
#     subMatirx = torch.split(eigenvalue, 2, dim=1)
#     K = torch.zeros((len(subMatirx) * 2, len(subMatirx) * 2))
#     # for i in range(K.shape[0]):
#     for j in range(len(subMatirx)):
#         element_1 = torch.exp(subMatirx[j][0][0] * deltaT) * torch.cos(subMatirx[j][0][1] * deltaT)
#         element_2 = torch.exp(subMatirx[j][0][0] * deltaT) * torch.sin(subMatirx[j][0][1] * deltaT)
#         K[j * 2][j * 2] = element_1
#         K[j * 2][j * 2 + 1] = - element_2
#         K[j * 2 + 1][j * 2] = element_2
#         K[j * 2 + 1][j * 2 + 1] = element_1
#
#     return K


def reconstruct(data, ae):
    return ae.decode(ae.encode(data))


def predict(data, index, nets):
    b_X = data[index, :DIM_STATE]

    b_U_set = []
    for step in range(PRED_LENGTH):
        b_U_set.append(data[index + step][:, -DIM_CTRL:])

    b_Z = nets.encode(b_X)  # (64, 100)
    b_Z_pred = b_Z.unsqueeze(dim=-1)
    for step in range(PRED_LENGTH):
        b_Z_pred = A @ b_Z_pred + net_b(b_U_set[step]).unsqueeze(dim=-1)

    b_X_pred = ae.decode(b_Z_pred.to(DEVICE).squeeze(dim=-1))  # (64, 50)

    return b_X_pred


def linear_trans2(K: torch.Tensor, y: torch.Tensor, u_set: list):
    assert K.shape[0] == y.shape[0] and PRED_LENGTH >= 1

    y = torch.unsqueeze(y, dim=-1)  # (64, 100, 1)
    A = K[:, :, :DIM_STATE * 2]  # (64, 100, 100)
    # B = K[:, :, -DIM_CTRL * 2:]  # (64, 100, 10)
    PSI = get_PSI(A)  # (64, 100n, 100)
    THETA = get_THETA(A, B)  # (64, 100n, 10n)
    U = torch.as_tensor(u_set, dtype=torch.float).permute((1, 0, 2))
    U = U.reshape(U.shape[0], -1).unsqueeze(dim=-1)  # (64, 10n, 1)

    y_hat = torch.matmul(PSI[:, -y.shape[1]:, :], y) + torch.matmul(THETA[:, -y.shape[1]:, :], U)
    # (64,100,1) = (64,100,100) * (64,100,1) + (64,100,10n) * (64,10n,1)

    return y_hat.reshape(y_hat.shape[0], -1)  # (64, 100)


def get_PSI(A):
    PSI = torch.empty((A.shape[0], A.shape[1] * PRED_LENGTH, A.shape[2]))
    last_A = torch.eye(A.shape[1])
    for i in range(PRED_LENGTH):
        res = torch.matmul(last_A, A)
        PSI[:, i * A.shape[1]: (i+1) * A.shape[1], :] = res
        last_A = res

    return PSI


def get_THETA(A, B):
    THETA = torch.zeros((A.shape[0], B.shape[1] * PRED_LENGTH, B.shape[2] * PRED_LENGTH))
    last_step = torch.tensor([])
    for i in range(PRED_LENGTH):
        if i > 0:
            last_step = torch.matmul(A, last_step)
        last_step = torch.cat((last_step, B), dim=2)
        THETA[:, i * B.shape[1]: (i+1) * B.shape[1], :(i+1) * B.shape[2]] = last_step

    return THETA
