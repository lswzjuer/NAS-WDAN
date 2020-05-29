import sys
import genotypes
from graphviz import Digraph
import os


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


def plot(genotype, filename):
  g = Digraph(
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



def create_all_imgs(save_path):
  '''
  :param save_path: ./cells
  :return:
  '''
  # change two list if you want to ad  new experments
  experments_list=["nas_search_layer7",
                   'stage0_nas_seach',
                   'stage1_nas_search',
                   'mixup_isic_traning'
                   ]
  genotypes_list=[
                  ['layer7_deepvision_doublechannel','layer7_doublechannel',
                   'layer7_deepvision_doublechannel_output2',
                   'layer7_doublechannel_output2'],
                  ['stage0_layer7_110epoch_double_final',
                   'stage0_layer7_110epoch_double_deep_final',
                   'stage0_layer7_110epoch_double_best',
                   'stage0_layer7_110epoch_double_deep_best'],
                ['stage1_layer9_110epoch_double_deep_final',
                 'stage1_layer9_110epoch_double_final',
                 'stage1_layer9_110epoch_deep_final',
                 'stage1_layer9_110epoch_final'],
                [
                    'alpha1_stage0_ep71',
                    'alpha0_stage0_ep60',
                    'alpha1_stage1_double_deep_ep200',
                    'alpha0_stage1_double_deep_ep200',
                    'alpha0_5_stage1_double_deep_ep80'
                ]
                ]

  assert os.path.exists(save_path),"the save path is not exists !"
  for i in range(len(experments_list)):
    dir_path=os.path.join(save_path,experments_list[i])
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)
    for j in range(len(genotypes_list[i])):
      experment_name=genotypes_list[i][j]
      try:
        genotype = eval('genotypes.{}'.format(experment_name))
      except AttributeError:
        print("{} is not specified in genotypes.py".format(experment_name))
      experment_path=os.path.join(dir_path,experment_name)
      if not os.path.exists(experment_path):
        os.mkdir(experment_path)
      plot(genotype.normal_down, os.path.join(experment_path,"normal_down"))
      plot(genotype.normal_up, os.path.join(experment_path,"normal_up"))
      plot(genotype.normal_normal, os.path.join(experment_path,"normal_normal"))


if __name__ == '__main__':
  import argparse
  parser=argparse.ArgumentParser("visiualize cell struct")
  parser.add_argument("--data",type=str,default="./cells",help="save path of experments imgs")
  args=parser.parse_args()
  create_all_imgs(args.data)


