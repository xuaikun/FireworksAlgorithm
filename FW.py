#encoding: utf-8
import random
import numpy as np
import math
txtSaveFlag = 0
'''
输入：种群规模，节点数
输出：种群情况
'''
def pop(size, NodeNumber, id_i):
    if id_i == 0:
        # PopSum = np.loadtxt('PopSum.txt', dtype=np.int)
        if txtSaveFlag == 0:
            PopSum = []
            for i in range(size):
                OnePop = random.sample(range(0, NodeNumber), NodeNumber)
                PopSumTemp = []
                for j in range(NodeNumber):
                    PopSumTemp.append(OnePop[j])
                PopSumTemp.append(NodeNumber)
                PopSum.append(PopSumTemp)
            print(PopSum)
            np.savetxt('150PopSum.txt', PopSum, fmt='%d')
    elif id_i == 1:
        # PopSum = np.loadtxt('PopSum1.txt', dtype=np.int)
        if txtSaveFlag == 0:
            PopSum = []
            for i in range(size):
                OnePop = random.sample(range(0, NodeNumber), NodeNumber)
                PopSumTemp = []
                for j in range(NodeNumber):
                    PopSumTemp.append(OnePop[j])
                PopSumTemp.append(NodeNumber)
                PopSum.append(PopSumTemp)
            print(PopSum)
            np.savetxt('150PopSum1.txt', PopSum, fmt='%d')
    elif id_i == 2:
        # PopSum = np.loadtxt('PopSum2.txt', dtype=np.int)
        if txtSaveFlag == 0:
            PopSum = []
            for i in range(size):
                OnePop = random.sample(range(0, NodeNumber), NodeNumber)
                PopSumTemp = []
                for j in range(NodeNumber):
                    PopSumTemp.append(OnePop[j])
                PopSumTemp.append(NodeNumber)
                PopSum.append(PopSumTemp)
            print(PopSum)
            np.savetxt('150PopSum2.txt', PopSum, fmt='%d')
    elif id_i == 3:
        # PopSum = np.loadtxt('PopSum3.txt', dtype=np.int)
        if txtSaveFlag == 0:
            PopSum = []
            for i in range(size):
                OnePop = random.sample(range(0, NodeNumber), NodeNumber)
                PopSumTemp = []
                for j in range(NodeNumber):
                    PopSumTemp.append(OnePop[j])
                PopSumTemp.append(NodeNumber)
                PopSum.append(PopSumTemp)
            print(PopSum)
            np.savetxt('150PopSum3.txt', PopSum, fmt='%d')

    print("PopSum =", PopSum)
    return PopSum


'''
输入：节点之间的距离，种群
输出：节点适度函数
'''
def getFitness(distance, popsum):
    # print("distance =", distance)
    # 使用np.zeros()时， NumSum每次初始化为0了,其中的值不可能为随机值
    # 之前使用np.empty()初始化NumSum导致矩阵中产生随机值，使得计算结果不正确
    # 用于保存每个点的适度值
    NumSum = np.zeros([1, len(popsum)], dtype=float)
    for i in range(len(popsum)):
        # 从第0位开始
        for j in range(0, len(popsum[0]) - 1):
            NumSum[0][i] = NumSum[0][i] + distance[popsum[i][j]][popsum[i][j + 1]]
        # 从最后一个节点返回仓库
        NumSum[0][i] = NumSum[0][i] + distance[popsum[i][0]][popsum[i][len(popsum[0]) - 1]]
    # print("NumSum[0] =", NumSum[0])
    return NumSum[0]


'''
输入：种群数，适度函数
输出：种群熵
'''
def PE(size, fitness):
    # 最大适度值
    f_max = fitness[np.argmax(fitness)]
    # 最小适度值
    f_min = fitness[np.argmin(fitness)]
    # 最值之间的间隔长度
    Lambda = float(f_max - f_min)
    c_i = []
    for i in range(size):
        down = float(f_min + Lambda * (i - 1)/size)
        up = float(f_min + Lambda * i/size)
        m_i = 0
        for j in range(size):
            if down <= fitness[j] < up:
                m_i += 1
        c_i.append(float(m_i/size))
    # print("c_i =", c_i)
    pe = 0
    for i in range(size):
        if c_i[i] == 0.0:
            # log括号中的值得大于0
            pe -= c_i[i] * math.log(1e-27)
        else:
            pe -= c_i[i] * math.log(c_i[i])
    # print("pe =", pe)
    return pe, f_max, f_min

'''
输入：节点距离矩阵，路径
输出：路径长度
'''
def getPathlength(distance, path):
    # print("getPathlengt path =", path)
    pathdistance = 0.0
    for i in range(len(path) - 1):
        pathdistance += distance[path[i]][path[i+1]]
        # 从最后一个节点返回仓库
    pathdistance += distance[path[0]][path[len(path) - 1]]
    # print("pathdistance =", pathdistance)
    return round(pathdistance, 2)

# 路径反转
# 获取随机的起始点还有中间的反转后的路径
def get_reverse_path(path):
    start = random.randint(0, len(path) - 2)
    while True:
        end = random.randint(0, len(path) - 2)
        if np.abs(start - end) > 1:
            break

    if start > end:
        path[end: start + 1] = path[end: start + 1][::-1]
        return path
    else:
        path[start: end + 1] = path[start: end + 1][::-1]
        return path

# nature selection wrt pop's fitness
# 使用的是随机抽样，不过不是等概率的抽样，而是根据适度值的大小与适度值之和进行比较
# 通过群体的适度函数进行自然选择操作
# 本程序当前适用于求最小值的类型，若日后遇到求最大值的，则对最大值的处理就进行取消
def select(popDNA, fitness, pOP_SIZE, popsize):
    #  np.random.choice(a, size=3, replace=False, p=None) 表示抽样选择
    #  从a（a = np.arange(a)-->a个随机数）中以p的概率选择size个不相同的数，replace=False 表示抽出后不放回，表示不会出现重复数据
    #  replace=True表示抽出后继续放回，会出现重复数据， p=None 表示 概率一致性， p =[0.1,0, 0.3, 0.6, 0]选中每一个数的概率不相同
    #  返回的结果为选中的数据在a中的位置【有size个id】

    # {
    # 	最求最小值的处理
    # 	idx = np.random.choice(np.arange(pOP_SIZE), size=pOP_SIZE, replace=True,
    #  	                     p = fitness/fitness.sum())
    # }

    # {
    # 求最大值的处理

    # 定义最大fitness值
    maxfitness = np.zeros([1, len(fitness)], dtype=float)
    # 只取第一个元素的值，并且需要＋1e-3加个小的数不至于新的fitness值出现0-》导致概率p等于0(错误)
    # 1e-3 = 1X10^-3 = 1/1000 = 0.001
    maxfitness[:] = fitness[np.argmax(fitness)] + 1e-3
    # 选择的概率，目前选择概率=是当前节点的适应度值/适应度总和，本文实验适应度值是选择低的
    # 导致适应度低的节点没有选择😂，与实际结果相反了，

    # 解决方法，用最大适应度 - 当前适应度/（最大适应度 - 当前适应度）总和
    # 产生的点附近还会有更多的点（昨晚情况相反，适应度低的点，周围没什么点）
    # 修改fitness，得到新的fitness值
    fitness = maxfitness[0] - fitness
    # p 为更新后的概率
    # p = fitness/fitness.sum()
    # 日后如果遇到求最大值的就不需要以上处理
    # print("np.arange(pOP_SIZE) =", np.arange(pOP_SIZE))
    idx = np.random.choice(np.arange(pOP_SIZE), size=popsize, replace=True,
                           p=fitness / fitness.sum())
    # }
    # print "idx =", idx
    # print "popDNA[idx] =\n", popDNA[idx]
    # 选出暂时存活的个体
    return popDNA[idx]

'''

'''
def FWA(distance, popsum, size):
    nc = 1
    # NCmax = 3600  # 最大迭代数
    NCmax = 3600
    PEList = []
    PETimeList = []
    m = 50
    # 论文中得出的
    sita = 0.8
    while nc < NCmax:
        print("第", nc, "次循环")
        # 获得节点适度函数
        fitness = getFitness(distance, popsum)
        pe, f_max, f_min = PE(size, fitness)
        PEList.append(pe)
        PETimeList.append(nc)
        sum_y_f = 0.0
        for i in range(size):
            sum_y_f += f_max - fitness[i]
        # print("sum_y_f =", sum_y_f)
        s_i =[]
        for i in range(size):
            s_i.append(int(50 * (((f_max - fitness[i] + 1e-27))/(sum_y_f + 1e-27))))
        # print("s_i =", s_i)
        # 路径备份
        popsumbackup = np.copy(popsum)
        # 每个烟花都需要操作，并且按火星数进行2-opt操作,检测正常！！！
        for i in range(size):
            # 对于第i个烟花产生的火星数量
            # popsum[i]
            # 根据火星个数改变当前烟花的种类
            # print("Init popsum[i] =", popsum[i])
            for j in range(s_i[i]):
                # 对当前种群做一个备份
                temp = np.copy(popsum[i])
                # print("Temp popsum[i] =", popsum[i])
                # 旧路径
                oldpath = getPathlength(distance, popsum[i])
                # 2-opt操作，保留较好的路径
                path = get_reverse_path(popsum[i])
                # print("Charge popsum[i] =", popsum[i])
                # 更新后的路径
                # print("path =", path)
                newpath = getPathlength(distance, path)
                # print("oldpath =", oldpath)
                # print("newpath =", newpath)
                # 新路径比旧路径长，保留之前的路径
                if newpath > oldpath:
                    # 将修改后的路径进行复原
                    popsum[i] = temp
                # 次优解的接受率
                try:
                    p_a = math.exp(-((newpath - oldpath + 1e-27)/(newpath - f_max + 1e-27)) * sita)
                except OverflowError:
                    p_a = float('inf')
                # p_a = 0.2
                # print("p_a =", p_a)
                # 在一定概率下接收次优的路径
                # print("final befor path =", popsum[i])
                rand = np.random.rand()
                # print("rand =", rand)
                if p_a > rand:
                    # 将修改后的路径进行复原
                    popsum[i] = temp
                # print("final after path =", popsum[i])

            # print("final popsum[i] =", popsum[i])

        # print("popsumbackup =", popsumbackup)
        # print("popsum =", popsum)
        # 可获得新的popsum，种群
        # 路径备份
        popsumbackup = np.copy(popsum)
        # print("popsumbackup =", popsumbackup)
        # 变异操作
        for i in range(size):
            # 2-opt操作，产生更多路径
            path = get_reverse_path(popsum[i])
            # print("path =", path)
            popsumbackup = np.r_[popsumbackup, [path]]
        # print("popsumbackup", popsumbackup)

        # 新的种群集合
        # 获取新种群的适度值
        fitness = getFitness(distance, popsumbackup)
        SolveNumber = np.argmin(fitness)  # 最小适度函数对应的第几个解决方案
        FitnessValue = fitness[np.argmin(fitness)]  # 最小适度值的值
        Solve = popsumbackup[np.argmin(fitness)]  # 最小适度函数对应的详细解决方案
        pe_new, f_max, f_min = PE(len(popsumbackup), fitness)
        # 删掉最优的路径
        popsumbackup = np.delete(popsumbackup, SolveNumber, axis=0)
        # 删掉最优路径对应的适度值
        fitness = np.delete(fitness, SolveNumber)
        # 从剩余的多条路径中选择N-1条路径
        popsum = select(popsumbackup, fitness, len(popsumbackup), size - 1)
        # 合并最优路径，获得一次更新的路径，更新种群popsum
        popsum = np.r_[popsum, [Solve]]
        # print("final popsum =", popsum)
        efsl = 0.65
        # sita = 0.8  ## 我觉得既然已经明确0.8最好，就选0.8就行，其它不用计算也行
        alpha = 1.4
        beta = 0.8
        detape = np.abs(pe - pe_new)
        if detape > efsl * pe:
            sita = sita * alpha
        if detape < (1 - efsl) * pe:
            sita = sita * beta
        # 迭代次数增加
        nc += 1
        # np,show()

        '''
        # np.argmin(a) 找出a的最小值索引
        # 获得解决方案
        SolveNumber = np.argmin(fitness)  # 最小适度函数对应的第几个解决方案
        FitnessValue = fitness[np.argmin(fitness)]  # 最小适度值的值
        Solve = PopSum[np.argmin(fitness), :]  # 最小适度函数对应的详细解决方案
        print("np.argmin(fitness) =", SolveNumber)
        print("min(fitness) =", FitnessValue)
        print("Most fitted xDNA: ", Solve)

        # 找到当前代的最小距离和
        bestfitness = fitness[np.argmin(fitness)]
        avefitness = fitness.sum() / len(fitness)
        bestDNA = np.argmin(fitness)

        print("bestfitness =", bestfitness)
        print("avefitness =", avefitness)
        print("bestDNA =", bestDNA)

        # 种群熵
        pe_1 = 0  ###     记得改改哦
        efsl = 0.65
        sita = 0.8  ## 我觉得既然已经明确0.8最好，就选0.8就行，其它不用计算也行
        alpha = 1.4
        beta = 0.8
        detape = np.abs(pe_1 - pe)
        if detape > efsl * pe_1:
            sita = sita * alpha
        if detape < (1 - efsl) * pe_1:
            sita = sita * beta
        pe_1 = pe
        '''
    fitness = getFitness(distance, popsum)
    print("fitness =", fitness)
    # print("popsum =", popsum)
    SolveNumber = np.argmin(fitness)  # 最小适度函数对应的第几个解决方案
    popsum = popsum[SolveNumber]
    TourList = []
    TourList.append(len(popsum) - 1)
    for i in range(len(popsum)):
        TourList.append(popsum[i])
    print("TourList =", TourList)
    # print("len(TourList) =", len(TourList))

    # 251.91 -->3600
    # 1220.3 -->7200
    # 1183.38 -->10800
    # 1122.17-->15000
    # 1194.6 -->20000
    # 1148.8 -->100000
    return TourList, [PEList, PETimeList], fitness


def FW_main(anchorNumber, id_i):
    # 种群大小
    Size = 10
    PopSum = pop(Size, anchorNumber, id_i)
    Distance = np.loadtxt("DistanceAC_SS.txt")
    popsum, PEValue, fitness = FWA(Distance, PopSum, Size)
    # 生成的路径可以保存起来，之后可以直接使用
    if id_i == 0:
        np.savetxt("Tour_L.txt", popsum, fmt='%d')
        np.savetxt("Tour_length.txt", fitness, fmt='%0.2f')
    elif id_i == 1:
        np.savetxt("Tour_L1.txt", popsum, fmt='%d')
        np.savetxt("Tour_length1.txt", fitness, fmt='%0.2f')
    elif id_i == 2:
        np.savetxt("Tour_L2.txt", popsum, fmt='%d')
        np.savetxt("Tour_length2.txt", fitness, fmt='%0.2f')
    elif id_i == 3:
        np.savetxt("Tour_L3.txt", popsum, fmt='%d')
        np.savetxt("Tour_length3.txt", fitness, fmt='%0.2f')
    return popsum, PEValue, fitness
