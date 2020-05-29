import sys
import genotypes
from graphviz import Digraph
import os


def plot(genotype, filename):
  g = Digraph(
      format='svg',
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
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")
  g.render(filename, view=False)




def create_all_imgs(save_path):
  '''
  :param save_path: ./
  :return:
  '''
  # change two list if you want to ad  new experments
  experments_list=["single_layer7",
                   'stage1_end'
                   ]
  genotypes_list=[
                  ['nodouble_deep_w10_layer7end','nodouble_deep_drop02_layer7end','nodouble_deep_w20_drop02_layer7end'],
                  ['stage1_nodouble_deep_a30_w10']
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
  parser.add_argument("--save_path",type=str,default="./cells_refuge",help="save path of experments imgs")
  args=parser.parse_args()
  create_all_imgs(args.save_path)


