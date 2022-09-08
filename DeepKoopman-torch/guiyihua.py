import numpy as np

d=np.load('./data/NGSIM/NGSIM_US101_Density_Data.npy')
d_min=d.min()
d_nor=d-d_min
d_max=d_nor.max()
d_nor=(d_nor/d_max*2-1)*2
np.save("data/NGSIM/NGSIM_US101_Density_Data_guiyihua.npy", d_nor)
print(d_nor)