# 处理图片
# 1.如何将图片转化为文本格式
# 使用pillow模块  安装pip3 install pillow
# 先将所有图片转为固定宽高，比如32*32，然后再转化为文本
# import pillow as pil
from PIL import Image  # 注意这里的Image的I是大写

ima = Image.open("D:/应用软件/python/class6/weixin.jpg")
fh = open("D:/应用软件/python/class6/weixin.txt", "a")  # 创建文本文件，用来保存0,1文件
ima.save("D:/应用软件/python/class6/weixincopy.bmp")  # 可直接保存为另一个图片
print(ima.size)  # 获取图片的长和宽
width = ima.size[0]
height = ima.size[1]
# k=ima.getpixel((1,9))  #获取x=1,y=9的像素值
# print(k)
for i in range(0, width):
    for j in range(0, height):
        color = ima.getpixel((i, j))  # 得到三通道值
        colorall = color[0] + color[1] + color[2]  # 计算三个通道总的像素值
        if (colorall == 0):  # 黑色
            fh.write("1")
        else:
            fh.write("0")
    fh.write("\n")  # 读完一行就换行
fh.close()


# 加载数据
def datatoarray(fname):  # 参数：文件名字
    arr = []
    # fh=open(fname,encoding='UTF-8')
    fh = open(fname, mode='r', encoding='UTF-8')  # 改了的
    for i in range(0, 32):
        thisline = fh.readline()
        # thisline=str(fh.readline())   也不行
        for j in range(0, 32):
            arr.append(int(thisline[j]))
    return arr


arr1 = datatoarray("D:/应用软件/python/class6/traindata/1_7.txt")  # 检查是否能够加载
print(arr1)  # 已验证，确实可以实现加载数据


# 1_7.txt若是打开过，肯会出现IndexError: string index out of range的错误，具体原因未知

# 建立一个函数取文件的前缀
def seplabel(fname):
    filestr = fname.split(".")[0]  # 分隔“.”,取第一个
    label = int(filestr.split("_")[0])  # 分隔“_”
    return label


# 建立训练数据
from os import listdir  # 是os模块下的


def traindata():
    labels = []  # 建立一个空的
    trainfile = listdir("D:/应用软件/python/class6/traindata")  # 得到一个文件夹下面的所有文件名
    num = len(trainfile)  # 列表trainfile有多少个文件，用len
    # 行长度32*32=1024（列），每一行存储一个文件
    # 用一个数组存储所有训练数据，行：文件总数，列：1024
    # zeros((2,5))  #numpy下的生成2行5列的数组
    trainarr = zeros((num, 1024))  # num个文件
    for i in range(0, num):  # 将真实数据放到trainarr中
        thisfname = trainfile[i]
        thislabel = seplabel(thisfname)  # 要取前缀
        labels.append(thislabel)  # 添加数据进去
        trainarr[i, :] = datatoarray("D:/应用软件/python/class6/traindata/" + thisfname)  # 当前目录下
        # trainarr[i,:]=datatoarray("traindata/"+thisfname)  #当前目录下
    return trainarr, labels


'''
#用测试数据调用KNN算法去测试门槛是否能够准确识别
def datatest():
    trainarr,labels=traindata()  #得到训练集和标签
    #testlist=listdir("D:/应用软件/python/class6/testdata")  #得到测试集
    testlist=listdir("testdata/")  #得到测试集
    tnum=len(testlist)
    for i in range(0,tnum):
        thistestfile=testlist[i]   #得到当前的测试文件
        #进行测试，首先加载为数组
        #testarr=datatoarray("D:/应用软件/python/class6/testdata/"+thistestfile)
        testarr=datatoarray("testdata/"+thistestfile)
        rknn=knn(3,testarr,trainarr,labels)
        print(rknn)

datatest() #调用函数   调用出错
'''

# 抽某一个测试文件出来进行试验
trainarr, labels = traindata()  # 得到训练集和标签
thistestfile = "6_50.txt"
testarr = datatoarray("D:/应用软件/python/class6/testdata/" + thistestfile)
rknn = knn(3, testarr, trainarr, labels)
print(rknn)