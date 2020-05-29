import imageio
import numpy as np
import os
import sys
from graphviz import Digraph
import os
import pickle
from PIL import Image


sys.path.append("../")
import genotypes

trans = {
    'none': "none",
    'identity': "identity",
    'cweight': "att_identity",
    'dil_conv': "atrous_conv",
    'dep_conv': "sep_conv",
    'shuffle_conv': "shuffle_conv",
    'conv':"conv",


    'avg_pool': "avg_pooling",
    'max_pool': "max_pooling",
    'down_cweight': "down_att_conv",
    'down_dil_conv': "donw_atrous_conv",
    'down_dep_conv': "down_sep_conv",
    'down_conv':"down_conv",


    'up_cweight': "up_att_conv",
    'up_dep_conv': "up_atrous_conv",
    'up_conv':"up_conv",
    'up_dil_conv': "up_sep_conv",
}


def plot(genotype, filename,title):
    '''
    :param genotype:
    :param filename:
    :param title:
    :return:
    '''
    g = Digraph(
        name="{}".format(title),
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')
    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=trans[op], fillcolor="gray")
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")
    g.render(filename, view=False)







def create_gif(file_list,save_path,duration=1):
    '''
    :param file_list:
    :param save_path:
    :return:
    '''
    """
     source:pic list 
     name ：file name 
     duration: time step 
    """
    frames = []
    for img in file_list:
        img=Image.open(img)
        img=img.resize((512,512), Image.NEAREST)
        frames.append(np.asarray(img))
    imageio.mimsave(save_path, frames, 'GIF', duration=duration)
    print("Finsh!")


def create_all_imgs(save_path):
    '''
    :param save_path:
    :return:
    '''
    assert os.path.exists(save_path),"the save path is not exists !"
    search_dirs= os.listdir(save_path)
    search_paths=[os.path.join(save_path,p) for p in search_dirs]
    stage0_path=None
    stage1_path=None
    for path in search_paths:
        files=os.listdir(path)
        for file in files:
            if "stage0_recoder_list" in file:
                stage0_path=os.path.join(path,file)
            elif "stage1_recoder_list" in file:
                stage1_path=os.path.join(path,file)

        # create stage0 dir
        if stage0_path:
            with open(stage0_path,"rb") as f:
                stage0=pickle.load(f)
            stage0_dir=os.path.join(path,"stage0")
            if not os.path.exists(stage0_dir):
                os.mkdir(stage0_dir)
            stage0_down=os.path.join(stage0_dir,"down")
            stage0_up=os.path.join(stage0_dir,"up")
            stage0_normal=os.path.join(stage0_dir,"normal")

            for tup in stage0:
                epoch,genotype,dice,jc=tup
                title="Epoch:{}  Dice:{:.4f}  Jc:{:.4f}".format(epoch,dice,jc)
                plot(genotype.normal_down, os.path.join(stage0_down,"{}".format(epoch)),title)
                plot(genotype.normal_up,  os.path.join(stage0_up,"{}".format(epoch)),title)
                plot(genotype.normal_normal, os.path.join(stage0_normal,"{}".format(epoch)),title)

            down_files=os.listdir(stage0_down)
            down_files=[file for file in down_files if "png" in file]
            down_files=sorted(down_files,key=lambda x: int(x.split(".")[0]))
            down_files_path=[os.path.join(stage0_down,file) for file in down_files]
            create_gif(down_files_path,os.path.join(stage0_dir,"down.gif"),1)

            up_files = os.listdir(stage0_up)
            up_files = [file for file in up_files if "png" in file]
            up_files = sorted(up_files, key=lambda x: int(x.split(".")[0]))
            up_files_path = [os.path.join(stage0_up, file) for file in up_files]
            create_gif(up_files_path, os.path.join(stage0_dir, "up.gif"), 1)

            normal_files = os.listdir(stage0_normal)
            normal_files = [file for file in normal_files if "png" in file]
            normal_files = sorted(normal_files, key=lambda x: int(x.split(".")[0]))
            normal_files_path = [os.path.join(stage0_normal, file) for file in normal_files]
            create_gif(normal_files_path, os.path.join(stage0_dir, "normal.gif"), 1)



        if stage1_path:
            with open(stage1_path,"rb") as f:
                stage1=pickle.load(f)
            stage1_dir=os.path.join(path,"stage1")
            stage1_down=os.path.join(stage1_dir,"down")
            stage1_up=os.path.join(stage1_dir,"up")
            stage1_normal=os.path.join(stage1_dir,"normal")
            if not os.path.exists(stage1_dir):
                os.mkdir(stage1_dir)
            for tup in stage1:
                epoch,genotype,dice,jc=tup
                title="Epoch:{}  Dice:{}  Jc:{}".format(epoch,dice,jc)
                plot(genotype.normal_down, os.path.join(stage1_down,"{}".format(epoch)),title)
                plot(genotype.normal_up,  os.path.join(stage1_up,"{}".format(epoch)),title)
                plot(genotype.normal_normal, os.path.join(stage1_normal,"{}".format(epoch)),title)

            down_files = os.listdir(stage1_down)
            down_files = [file for file in down_files if "png" in file]
            down_files = sorted(down_files, key=lambda x: int(x.split(".")[0]))
            down_files_path = [os.path.join(stage1_down, file) for file in down_files]
            create_gif(down_files_path, os.path.join(stage1_dir, "down.gif"), 1)

            up_files = os.listdir(stage1_up)
            up_files = [file for file in up_files if "png" in file]
            up_files = sorted(up_files, key=lambda x: int(x.split(".")[0]))
            up_files_path = [os.path.join(stage1_up, file) for file in up_files]
            create_gif(up_files_path, os.path.join(stage1_dir, "up.gif"), 1)

            normal_files = os.listdir(stage1_normal)
            normal_files = [file for file in normal_files if "png" in file]
            normal_files = sorted(normal_files, key=lambda x: int(x.split(".")[0]))
            normal_files_path = [os.path.join(stage1_normal, file) for file in normal_files]
            create_gif(normal_files_path, os.path.join(stage1_dir, "normal.gif"), 1)



if __name__=="__main__":
    search_path=r'E:\segmentation\Image_Segmentation\nas_search_unet\search_exp\Nas_Search_Unet\total'
    create_all_imgs(search_path)