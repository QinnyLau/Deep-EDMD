import numpy as np
import torch
test1=np.load("./data/NGSIM/NGSIM_US101_Density_Data_guiyihua.npy")
t=np.load("./data/NGSIM/NGSIM_US101_Density_Data.npy")
test1=test1.T
# tensor = tensor.to(torch.float32)
test=torch.tensor(test1)
data=[]
nets=torch.load("./saved/IDM_0902-22-17.pth")
nets = nets.double()
data_pred,data_in_high_dimension_pred = nets.predict(test)
data_recon=nets.recon(test)
print(data_pred)
print(data_recon)
data=(data_pred.detach())
data1=(data_recon.detach())
d=(data/2+1)/2*t.max()+t.min()
d1=(data1/2+1)/2*t.max()+t.min()
print(d)
print(d1)
np.savetxt("log1/density_pre.csv", d, fmt='%f', delimiter=',')
np.savetxt("log1/density_rec.csv", d1, fmt='%f', delimiter=',')