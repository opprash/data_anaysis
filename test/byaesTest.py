class Bayes:  # 定义一个贝叶斯类
    def __init__(self):  # 对类进行初始化
        self.length = -1  # 初始化变量长度
        self.labelcount = dict()  # 类别标签，并用空字典存储
        self.vectorcount = dict()  # 向量

    def fit(self, dataSet: list, labels: list):  # 训练方法，数据集，标签集 用：list指定类别
        if (len(dataSet) != len(labels)):  # 训练方法中，数据集与标签集的长度应该一致
            raise ValueError("您输入的测试数组与类别数组长度不一致")  # 提示出错，用raise引发异常
        self.length = len(dataSet[0])  # 定义长度 测试数据中任一维的长度，就是它特征值的长度
        labelsnum = len(labels)  # 类别所有的数量
        norlabel = set(labels)  # 不重复类别的数量 通过集合set方法出去重复的
        for item in norlabels:  # 依次遍历各个类别
            thislabel = item  # 当前类别占总类别的比例
            labelcount[thislabel] = labels.count(thislabel) / labelsnum
        for vector, label in zip(dataSet, labels):  # 遍历测试数据的类别和向量
            if (label not in vectorcount):  # label不在里面，则对其进行初始化
                self.vectorcount[label] = []
            self.vectorcount[label].append(vector)
        print("训练结束")
        return self

    def btest(self, TestData, labelSet):  # 将当前的测试数据、对应的类别集合输入
        if (self.length == -1):
            raise ValueError("您还没有进行训练，请先训练")
        # 计算testdata分别为各个类别的概率
        lbDict = dict()
        for thislb in labelSet:  # 依次遍历各个类别
            p = 1  # 后面公式用
            alllabel = self.labelcount[thislb]  # 统计当前类别占总类别的比例
            allvector = self.vectorcount[thislb]
            vnum = len(allvector)
            allvector = numpy.array(allvector).T
            for index in range(0, len(TestData)):
                vector = list(allvector[index])
                p *= vector.count(TestData[index]) / vnum  # 计算当前特征的概率
            lbDict[thislb] = p * alllabel
        thisislabel = sorted(lbDict, key=lambda x: lbDict[x], reverse=True)[0]  # 从大到小排序reverse=True
        return thisislabel


if __name__ == "__main__":
    by1 = Bayes()
