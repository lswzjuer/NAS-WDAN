import numpy as np 
from collections import defaultdict
import matplotlib.pyplot as plt

# data = np.array([['Model', 'Unet', 'Unet++ /wo', 'Att-Unet', 'R2t-Unet/t=3',
#         'Att-R2t-Unet', 'MultiRes-Unet', 'V1', 'Slim-V1', 'V2',
#         'Slim-V2', 'V3', 'Slim-V3', 'V4', 'Slim-V4'],
#        ['Dice', '83.8', '81.91', '83.93', '80.69', '80.8', '82.81',
#         '84.74', '85.49', '84.69', '83.83', '83.91', '83.96', '83.15',
#         '83.09'],
#        ['P', '34.53', '36.63', '34.89', '39.09', '39.44', '34.84',
#         '8.11', '7.7', '12.8', '12.6', '0.41', '0.371', '0.51', '0.487'],
#        ['CPU Time', '1.978', '19.53', '2.02', '24.21', '27.22', '10.41',
#         '1.4', '1.07', '1.33', '1.143', '0.78', '0.698', '0.92', '0.88']])

data = np.array([['Model', 'Unet', 'Unet++ /wo', 'Att-Unet', 'R2t-Unet/t=3',
         'MultiRes-Unet', 'V1', 'V2',
        'V3', 'V4'],
       ['Dice', '86.47', '86.44', '87.22', '81.97', '85.57',
        '87.73', '87.14',  '86.93',  '85.99',
        ],
       ['P', '34.53', '36.63', '34.89', '39.09', '34.84',
        '8.11', '12.8','0.41', '0.51'],
       ['CPU Time', '4.28', '14.12', '4.415', '14.87', '7.67',
        '1.402', '1.335', '0.782', '0.923']])



result = defaultdict(np.ndarray)
result[data[0][0]] = data[0][1:]
for i in range(1,4):
    l = []
    for j in range(len(data[i][1:])):
        t = float(data[i][1:][j])
        l.append(t)
    result[data[i][0]] = l

 
plt.figure(figsize=(10,8))#设置画布的尺寸
# plt.title('Examples of scatter plots',fontsize=20)#标题，并设定字号大小
plt.xlabel(u'CPU Time (s)',fontsize=14)#设置x轴，并设定字号大小
plt.ylabel(u'Dice (%)',fontsize=14)#设置y轴，并设定字号大小
plt.grid() # 设置网格
plt.xlim(0,16) # 设置横坐标范围
plt.ylim(80,90) # 设置纵坐标范围
#渐变色
# tab20c 出自: https://matplotlib.org/examples/color/colormaps_reference.html
cValue_2 = result['P'] # 颜色表示数值
cm = plt.cm.get_cmap('tab20c') # Blues
plt.scatter(result['CPU Time'],result['Dice'],c = cValue_2, s=500, marker='.', cmap=cm) # s 设置大小 marker='.' or '*'

# 添加数据标签
for a,b,c in zip(result['CPU Time'],result['Dice'],result['Model']):
    plt.text(a, b+0.05, c, ha='center', va= 'bottom',fontsize=10)

# 设置colorbar的标签字体及其大小
font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 15,
        }

plt.colorbar().set_label(u'Params (M)',fontdict=font)
plt.savefig(r"C:\Users\rileyliu\Desktop\TMI论文\ourpaper\figures\compare")

# cb=plt.colorbar() #h
# cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
# font = {'family' : 'serif',
#         'color'  : 'darkred',
#         'weight' : 'normal',
#         'size'   : 16,
#         }
# cb.set_label('colorbar',fontdict=font) #设置colorbar的标签字体及其大小

#指定点的颜色的序列
# cValue_1 = ['r','c','g','b','r','y','g','b','m']
# plt.scatter(result['CPU Time'],result['Dice'],c = cValue_1, s=100, marker='o')
 