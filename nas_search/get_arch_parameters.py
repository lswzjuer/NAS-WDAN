import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
import os







# model_dir='./search_exp/Nas_Search_Unet/total'
# dir_list=os.listdir(model_dir)
# dir_paths=[os.path.join(model_dir,i) for i in dir_list]
# stage_model_dir=[os.path.join(i,'stage_1_model',"checkpoint.pth.tar") for i in dir_paths]
#
# for model_index in range(len(stage_model_dir)):
#     ckpt=torch.load(stage_model_dir[model_index],map_location='cpu')
#     alphas_network=ckpt['alphas_dict']['alphas_network']
#     alphas_network=F.softmax(alphas_network, dim=-1).detach().numpy()
#     alphas_network_path=os.path.join(dir_paths[model_index],"alphas_network.pkl")
#     with open(alphas_network_path,"wb") as f:
#         pickle.dump(alphas_network,f)



sp_model_dir='./search_exp/Nas_Search_Unet/isic2018/deepsupervision'
alpha_path=os.path.join(sp_model_dir,"alphas_network.pkl")
with open(alpha_path,"rb") as f :
    alphas_network=pickle.load(f)

output_node=[[2,0],[3,1],[4,0],[4,2],[5,1],[5,3],[6,0],[6,2],[7,1],[8,0]]
for coord in output_node:
    print("coor:{}".format(coord))
    print("     {}".format(alphas_network[coord[0]][coord[1]]))





