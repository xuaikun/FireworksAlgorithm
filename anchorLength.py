#coding:utf-8
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from FW import *
import os
txtSaveFlag = 1

# 测试标志
testFlag = 1
# 充电范围为7.5m,记得使用浮点数
# D = 15.0
# 充电范围为2.7m，作为正方形的对角线，利用勾股定理得到正方形边长为1.92m记得使用浮点数~1.91m
D = 3.82
# MD充电率5W
U = 1.0
'''
输入：节点之间的距离矩阵
输出：每个节点充电接收率
# Multi-Node Wireless Energy Charging in Sensor Networks 论文中出现了该计算公式
'''
def U_i_j(distance):
    u_i_j = (-0.0958*distance*distance - 0.0377*distance + 1) * U
    return u_i_j


'''
输入：mat数据
输出：解析mat数据
'''
def loadmatData(matData):
    # 导出mat数据
    data = scio.loadmat(matData)
    if testFlag == False:
        print("All Data =", data)
    # 两两节点之间的距离
    Distance = data['Distance']
    # 节点数量
    NodesNumber = data['nodesnumber']
    NodesNumber = NodesNumber[0][0]
    NodeLabel = []
    # 节点的label
    for i in range(NodesNumber):
        NodeLabel.append(i)
    if testFlag == False:
        print("NodeLabel =", NodeLabel)
    # 节点之间的路由情况
    Routing = data['routing']
    # 节点感知距离
    SenseDistance = data['sensedistance']
    # 节点坐标（其实暗含各节点的标号）
    SequenceNumber = data['sequencenumber']
    NodeCoordinatesX = SequenceNumber[0]
    NodeCoordinatesY = SequenceNumber[1]
    if testFlag == False:
        print("NodeCoordinatesX =", NodeCoordinatesX, "\n NodeCoordinatesY =",  NodeCoordinatesY)
    # 运行区域的边界(单位为m)
    xSlide = data['xSlide']
    ySlide = data['ySlide']
    return NodeLabel, NodeCoordinatesX, NodeCoordinatesY, Distance, NodesNumber, Routing, SenseDistance, SequenceNumber, xSlide, ySlide

'''
输入:节点标号,节点坐标
输出:分区集合,锚点列表
'''
def AnchorInitSelect(nodeLabel, CoordinateX, CoordinateY, maxGridNumber):
    GridLabel = []
    # 最少需要多少方格将边长切分
    # print("nodeLabel =", nodeLabel)
    if testFlag == 0:
        print("maxGridNumber =", maxGridNumber)
    for i in range(maxGridNumber*(maxGridNumber)):
        GridLabel.append(i)
    if testFlag == 0:
        print("len(GridLabel)= ", len(GridLabel))
    GridNode = np.zeros([1, len(GridLabel)], dtype=list)
    # print("len(nodeLabel) =", len(nodeLabel))
    # 每次只是操作一个小分区，因此节点数量等于当前分区的数量
    for i in range(len(nodeLabel)):
        # 反推节点所在网格号
        # print("i =", i)
        NL = nodeLabel[i]
        x = round((CoordinateX[NL]-(D/2.0))/D)
        y = round((CoordinateY[NL]-(D/2.0))/D)
        if testFlag == 0:
            print("int(x+maxGridNumber*y)= ", int(x+maxGridNumber*y))
            print("GridLabel", GridLabel)
        temp = []
        valuetemp = GridNode[0][int(x+maxGridNumber*y)]
        if testFlag == 0:
            print("***********valuetemp =", valuetemp)
        if valuetemp != 0:
            if testFlag == 0:
                print("len(valuetemp) =", len(valuetemp))
            for j in range(len(valuetemp)):
                temp.append(valuetemp[j])
            if testFlag == 0:
                print("valuetemp =", valuetemp)
        temp.append(nodeLabel[i])
        if testFlag == 0:
            print("temp =", temp)
        GridNode[0][int(x + maxGridNumber * y)] = temp
        if testFlag == 0:
            print("GridNode =", GridNode)
    if testFlag == 0:
        print("over GridNode =", GridNode)
    # 保持存在传感器节点的单元格
    CellLabel = []
    for i in range(maxGridNumber*maxGridNumber):
        if GridNode[0][i] != 0:
            CellLabel.append(i)
    if testFlag == 0:
        print("CellLabel =", CellLabel)
    # 分区集合,同时也是为了知道锚点个数
    SensorSet = GridNode.ravel()[np.flatnonzero(GridNode)]
    if testFlag == 0:
        print("节点归属情况 =", SensorSet)
        print("锚点个数 =", len(SensorSet))
    anchorList = []
    AnchorCoordinateX = []
    AnchorCoordinateY = []
    for i in range(len(SensorSet)):
        # 保存锚点序号，自定义
        anchorList.append(i)
        # 每个区域c[x][y]的序号为:x + maxGridNumber*y
        # 知道序号后,反推x和y的值
        # 需要向下取整,得到当前列数
        y = math.floor(CellLabel[i] / maxGridNumber)
        x = CellLabel[i] - maxGridNumber * y
        # 计算中心节点的坐标,该坐标即为锚点坐标
        X = (x * D) + (D / 2.0)
        Y = (y * D) + (D / 2.0)
        # 保存锚点坐标，初始化为分区中心
        AnchorCoordinateX.append(X)
        AnchorCoordinateY.append(Y)

    if testFlag == 0:
        print("anchorList =", anchorList)
    return GridNode, CellLabel, anchorList, [AnchorCoordinateX, AnchorCoordinateY]

'''
锚点更新策略
输入:充电时间,数据收集时间,分区
输出:锚点坐标
'''
def APE_Strategy(gridNode, cellLabel, xData, yData, p_i, R_i, u_i_j, G, maxGridNumber):
    # 最少需要多少方格将边长切分

    # 更新后每个区域的锚点坐标
    AnchorCoordinateX = []
    AnchorCoordinateY = []
    # 从锚点列表中其对应的传感器序列
    i = 0
    for SensorList in gridNode[0][cellLabel]:
        if testFlag == 0:
            print("SensorList =", SensorList)
            print("cellLabel[i] =", cellLabel[i])
        # 仅包含一个节点,锚点坐标就是节点坐标
        if len(SensorList) == 1:
            if testFlag == 0:
                print("仅有一个节点")
            X = xData[SensorList[0]]
            Y = yData[SensorList[0]]
        # 包含节点数超过1
        else:
            # 这里用的是坐标求和,初始应该为0
            X = 0
            Y = 0
            # 考虑节点能量消耗率相同!!!!!!!!!；
            # 充电时间大于数据收集时间
            tc_temp = []
            tg_temp = []
            for sensorSet in range(len(SensorList)):
                # print("p_i[SensorList[sensorSet]] =", p_i[SensorList[sensorSet]])
                # print("u_i_j[SensorList[sensorSet]] =", u_i_j[0][SensorList[sensorSet]])
                tc_temp.append(p_i[SensorList[sensorSet]]/u_i_j[0][SensorList[sensorSet]])
                tg_temp.append(R_i[SensorList[sensorSet]]/G)
            # 在多个时间中取最大值
            tc = max(tc_temp)
            tg = max(tg_temp)
            if tc > tg:
                if testFlag == 0:
                    print("时间不符合需求")
                # 每个单元能量消耗率总和
                p_i_sum = 0.0
                # 统计每个传感器节点的消耗率
                for Sensorj in range(len(SensorList)):
                    # print("SensorList =", SensorList)
                    # print("Sensorj =", Sensorj)
                    # print("SensorList[Sensorj] =", SensorList[Sensorj])
                    p_i_sum += p_i[SensorList[Sensorj]]
                    # print("我在调整锚点这里")
                    # np,show()

                # print("p_i_sum =", p_i_sum)
                # 求得节点的权重
                # print("p_i[SensorList] =", p_i[SensorList])
                w_i = p_i[SensorList]/p_i_sum
                # print("w_i =", w_i)
                for l in range(len(SensorList)):
                    # 每个节点的坐标乘以权重,然后求和,即为更新后的锚点位置
                    X += w_i[l]*xData[SensorList[l]]
                    Y += w_i[l]*yData[SensorList[l]]
                # print("(X, Y) =", (X, Y))
                #np,show()
            # 锚点位置为区域中心
            else:
                if testFlag == 0:
                    print("时间符合需求")
                # 每个区域c[x][y]的序号为:x + maxGridNumber*y
                # 知道序号后,反推x和y的值
                # 需要向下取整,得到当前列数
                y = math.floor(cellLabel[i]/maxGridNumber)
                x = cellLabel[i] - maxGridNumber*y
                # 计算中心节点的坐标,该坐标即为锚点坐标
                X = (x * D) + (D/2.0)
                Y = (y * D) + (D/2.0)
        # 往后移动一位
        i += 1

        AnchorCoordinateX.append(X)
        AnchorCoordinateY.append(Y)
        if testFlag == 0:
            print("X =", X, "\nY =", Y)
    return [AnchorCoordinateX, AnchorCoordinateY]


'''
最小距离服务方法
输入：Sensor能量消耗率p_i，Sensor数据生成率R_i，Sensor最大性能E_max，Sensor最小能量E_min，MD移动速率v,MD充电率U，MD数据收集率G，MD移动功率P_T
输出：最优周期时间T，最佳移动路径L，目标值n_1
'''
def PE_FWA(gridNode, cellLabel, anchorCoordinate):
    # 服务站的坐标为(50，50)
    SS = [50.0, 50.0]
    # 最大迭代次数
    NC_max = 100
    # 候选集
    CS = gridNode[0][cellLabel]
    # print("CS =", CS)
    # 锚点的坐标，锚点数量为44个
    ACX = anchorCoordinate[0]
    ACY = anchorCoordinate[1]
    ACX.append(SS[0])
    ACY.append(SS[1])
    # 锚点与服务站的坐标合并到了一块
    AC_SSX = ACX
    AC_SSY = ACY
    # print("AC_SSX =", AC_SSX)
    # print("AC_SSY =", AC_SSY)
    # 计算锚点之间的距离和锚点到服务站的距离,共个节点
    # 距离已经保存，可以直接提取使用
    DistanceAC_SS = np.loadtxt('DistanceAC_SS.txt')
    if txtSaveFlag == 1:
        DistanceAC_SS = np.zeros([len(AC_SSX), len(AC_SSX)], dtype=float)
        for i in range(len(AC_SSX)):
            for j in range(i + 1):
                if i != j:
                    DistanceAC_SS[i][j] = math.sqrt(math.pow((AC_SSX[i] - AC_SSX[j]), 2) + math.pow((AC_SSY[i] - AC_SSY[j]), 2))
                    DistanceAC_SS[j][i] = DistanceAC_SS[i][j]
                else:
                    DistanceAC_SS[i][j] = float('inf')
        # 后期可以直接保存，便于后期使用
        np.savetxt('DistanceAC_SS.txt', DistanceAC_SS, fmt='%0.2f')

    # 共45行45列
    # print("DistanceAC_SS =", DistanceAC_SS)
    # print("cellLabel =", cellLabel)
    # 用来保存锚点的排序序号和节点标号
    cellIndex = []
    # 保存锚点排序序号，也是锚点初始序列
    Tour_LTemp = []
    for i in range(len(cellLabel)):
        Tour_LTemp.append(i)
        cellIndextemp = []
        cellIndextemp.append(i)
        cellIndextemp.append(cellLabel[i])
        cellIndex.append(cellIndextemp)
    # print("cellIndex =", cellIndex)
    Tour_L = []
    # 首先把服务站加入巡回路径
    Tour_L.append(len(cellLabel))
    while len(Tour_LTemp) != 0:
        # 初始化最小距离为无穷大
        # print("Tour_LTemp =", Tour_LTemp)
        # print("Tour_L =", Tour_L)
        minDistance = float('inf')
        minDistanceIndex = Tour_LTemp[0]
        # os.system("pause")
        for TourNodeIndex in range(len(Tour_LTemp)):
            if DistanceAC_SS[Tour_L[len(Tour_L) - 1]][Tour_LTemp[TourNodeIndex]] < minDistance:

                # 修改最小距离
                minDistance = DistanceAC_SS[Tour_L[len(Tour_L) - 1]][Tour_LTemp[TourNodeIndex]]
                # 当前距离较小的节点的标号
                minDistanceIndex = Tour_LTemp[TourNodeIndex]

        Tour_LTemp.remove(minDistanceIndex)
        Tour_L.append(minDistanceIndex)
    Tour_L.append(Tour_L[0])
    # print("Tour_L =", Tour_L)
    # print("len(Tour_L) =", len(Tour_L))

    return Tour_L, AC_SSX, AC_SSY

'''
初始图的展示
输入：相关数据
输出：展示图
'''
def PoltInit(gridNode, cellLabel, anchorList, anchorCoordinate, xData, yData):
    # 离散点的表示
    plt.scatter(xData, yData, c='b')
    plt.scatter(50, 50, c='black', marker='p')
    # print("poltInit anchorList =", anchorList)
    for anchorNum in range(len(anchorList)):
        plt.scatter(anchorCoordinate[0][anchorList[anchorNum]], anchorCoordinate[1][anchorList[anchorNum]], c='red', marker="*")
        plt.text(anchorCoordinate[0][anchorNum], anchorCoordinate[1][anchorNum], anchorNum, fontsize=12)
    # 提取所有包含传感器节点的分区及其详情
    sensorSet = gridNode[0][cellLabel]
    # print("sensorSet =", sensorSet)
    # 初始锚点与其覆盖节点的连线,形象展示
    for i in range(len(sensorSet)):
        # 每个点都要连接
        # print("sensorSet =", sensorSet)
        # npn,show()
        for j in range(0, len(sensorSet[i])):
            sensor = sensorSet[i][j]
            '''
            print("sensor =", sensor)
            print("xData[sensor] =", xData[sensor])
            print("yData[sensor] =", yData[sensor])
            print("anchorList[i] =", anchorList[i])
            print("anchorCoordinate[0][anchorList[i]] =", anchorCoordinate[0][anchorList[i]])
            print("anchorCoordinate[1][anchorList[i]] =", anchorCoordinate[1][anchorList[i]])
            '''
            plt.plot((anchorCoordinate[0][anchorList[i]], xData[sensor]), (anchorCoordinate[1][anchorList[i]], yData[sensor]), 'black')
            #plt.show()
            #np,show()
    # 网格划分
    X = []
    Y = []
    # 如果充电距离为2.7米时，int(xSlide/D) + 1改为20
    step = 2  # 便于将示意图展示完全
    for i in range(int(xSlide/D) + step):
        if i == 0:
            X.append(0)
            Y.append(0)
        else:
            X.append(((i)*D))
            Y.append(((i)*D))
    for i in range(int(xSlide/D) + step):
        Xandy = []
        Yandx = []
        for j in range(int(xSlide/D) + step):
            Xandy.append(X[i])
            Yandx.append(Y[j])
        if testFlag == False:
            print(Xandy, Yandx)
        # np,s(0)
        plt.plot(Xandy, Yandx, '-y')
    for i in range(int(xSlide/D) + step):
        Xandy = []
        Yandx = []
        for j in range(int(xSlide/D) + step):
            Xandy.append(Y[i])
            Yandx.append(X[j])
        if testFlag == False:
            print(Yandx, Xandy)
        # np,s(0)
        plt.plot(Yandx, Xandy, '-y')
    # 展示图
    plt.show()

'''
锚点优化后图的展示
输入:
输出:
'''
def PlotOptimal(gridNode, cellLabel, anchorCoordinate, xData, yData):
    # plt.scatter(anchorCoordinate[0],anchorCoordinate[1])
    # 离散点的表示
    plt.scatter(xData, yData, c='b')
    plt.scatter(50, 50, c='black', marker='p')
    plt.scatter(anchorCoordinate[0], anchorCoordinate[1], c='red', marker="*")
    for anchorNum in range(len(anchorCoordinate[0])):
        plt.text(anchorCoordinate[0][anchorNum], anchorCoordinate[1][anchorNum], anchorNum, fontsize=12)
    # 提取所有包含传感器节点的分区及其详情
    sensorSet = gridNode[0][cellLabel]
    if testFlag == 0:
        print("sensorSet =", sensorSet)
    # 锚点在序列中的索引
    m = 0
    # 初始锚点与其覆盖节点的连线,形象展示
    for i in range(len(sensorSet)):
        if testFlag == 0:
            print("anchorCoordinate[0][j] =", anchorCoordinate[0][m])
            print("xData[sensorSet[i]] =", xData[sensorSet[i]])
        for j in range(0, len(sensorSet[i])):
            sensor = sensorSet[i][j]
            plt.plot((anchorCoordinate[0][m], xData[sensor]), (anchorCoordinate[1][m], yData[sensor]), 'black')
        m += 1
    # 网格划分
    X = []
    Y = []
    # 如果充电距离为2.7米时，int(xSlide/D) + 1改为20
    step = 2  # 便于将示意图展示完全
    for i in range(int(xSlide / D) + step):
        if i == 0:
            X.append(0)
            Y.append(0)
        else:
            X.append(((i) * D))
            Y.append(((i) * D))
    for i in range(int(xSlide / D) + step):
        Xandy = []
        Yandx = []
        for j in range(int(xSlide / D) + step):
            Xandy.append(X[i])
            Yandx.append(Y[j])
        if testFlag == False:
            print(Xandy, Yandx)
        # np,s(0)
        plt.plot(Xandy, Yandx, '-y')
    for i in range(int(xSlide / D) + step):
        Xandy = []
        Yandx = []
        for j in range(int(xSlide / D) + step):
            Xandy.append(Y[i])
            Yandx.append(X[j])
        if testFlag == False:
            print(Yandx, Xandy)
        # np,s(0)
        plt.plot(Yandx, Xandy, '-y')
    # 展示图
    plt.show()

'''
展示形成的回路路径
'''
def PlotTour(tour_L, aC_SSX, aC_SSY, gridNode, cellLabel, anchorCoordinate, xData, yData):
    # 离散点的表示
    plt.scatter(xData, yData, c='b')

    plt.scatter(anchorCoordinate[0], anchorCoordinate[1], c='red', marker="*")
    for i in range(len(anchorCoordinate[0])):
        plt.text(anchorCoordinate[0][i], anchorCoordinate[1][i], i, fontsize=12)

    # 提取所有包含传感器节点的分区及其详情
    sensorSet = gridNode[0][cellLabel]
    if testFlag == 0:
        print("sensorSet =", sensorSet)
    # 锚点在序列中的索引
    m = 0
    # 初始锚点与其覆盖节点的连线,形象展示
    for i in range(len(sensorSet)):
        if testFlag == 0:
            print("anchorCoordinate[0][j] =", anchorCoordinate[0][m])
            print("xData[sensorSet[i]] =", xData[sensorSet[i]])
        for j in range(0, len(sensorSet[i])):
            sensor = sensorSet[i][j]
            plt.plot((anchorCoordinate[0][m], xData[sensor]), (anchorCoordinate[1][m], yData[sensor]), 'black')
        m += 1
    # 构建回路
    # Tour_L, aC_SSX, aC_SSY
    m = 0
    # 初始锚点与其覆盖节点的连线,形象展示
    for i in range(len(tour_L) - 1):
        plt.plot((aC_SSX[tour_L[i]], aC_SSX[tour_L[i + 1]]), (aC_SSY[tour_L[i]], aC_SSY[tour_L[i + 1]]), 'red')
        m += 1

    # 网格划分
    X = []
    Y = []
    # 如果充电距离为2.7米时，int(xSlide/D) + 1改为20
    step = 2  # 便于将示意图展示完全
    for i in range(int(xSlide / D) + step):
        if i == 0:
            X.append(0)
            Y.append(0)
        else:
            X.append(((i) * D))
            Y.append(((i) * D))
    for i in range(int(xSlide / D) + step):
        Xandy = []
        Yandx = []
        for j in range(int(xSlide / D) + step):
            Xandy.append(X[i])
            Yandx.append(Y[j])
        if testFlag == False:
            print(Xandy, Yandx)
        # np,s(0)
        plt.plot(Xandy, Yandx, '-y')
    for i in range(int(xSlide / D) + step):
        Xandy = []
        Yandx = []
        for j in range(int(xSlide / D) + step):
            Xandy.append(Y[i])
            Yandx.append(X[j])
        if testFlag == False:
            print(Yandx, Xandy)
        # np,s(0)
        plt.plot(Yandx, Xandy, '-y')
    # 展示图
    plt.scatter(50, 50, c='black', marker='p')
    plt.show()


if __name__ == '__main__':
    # 节点最大能量
    # E_max = 10800.0
    E_max = 50.0
    # 小车最大能量
    MD_E_MAX = 50e3
    # 小车数据收集率原来是bit/s-->节点数据传输到MD的传输率
    G = 50 * 8
    # 原始网络图
    # 导入文件
    dataFile = '300_node.mat'
    # 对网络进行分区
    dataFileArea = '300node_150LTSP.mat'
    NodeLabel, NodeCoordinatesX, NodeCoordinatesY, Distance, NodesNumber, Routing, SenseDistance, SequenceNumber, xSlide, ySlide = loadmatData(
        dataFile)
    if testFlag == 0:
        print('Distance =', Distance, 'NodesNumber =', NodesNumber)
        print('Routing =', Routing)
        print('SenseDistance =', SenseDistance, 'SequenceNumber =', SequenceNumber)
        print('xSlide =', xSlide, 'ySlide =', ySlide)
    # print("NodeCoordinatesX =", NodeCoordinatesX)
    # print("NodeCoordinatesY =", NodeCoordinatesY)
    # np,show()
    np.savetxt("Distance.txt", Distance, fmt='%0.2f')
    # 每行最大网格数量
    MaxGridNumber = math.ceil(xSlide / D)

    Areadata = scio.loadmat(dataFileArea)
    # 需要知道有多少个分区
    dataArea = Areadata['Area']
    # print("dataArea =", dataArea)
    # print("len(dataArea) =", len(dataArea))
    AreaNumber = dataArea[0]
    # print("AreaNumber =", AreaNumber)
    # print("len(AreaNumber) =", len(AreaNumber))
    for ID_i in range(len(AreaNumber)):
        # 这里即为一个小分区的值，包含节点序号，节点横纵坐标，以及节点的能量
        # AreaNumber[0]-->第一分区
        # AreaNumber[1]-->第二分区
        # AreaNumber[2]-->第三分区
        # AreaNumber[3]-->第四分区
        tempArea = AreaNumber[ID_i]
        # tempArea[0]-->节点序号序列
        # tempArea[1]-->节点横坐标序列
        # tempArea[2]-->节点纵坐标序列
        # tempArea[3]-->节点能量序列
        # print("tempArea[0] =", tempArea[0])
        # matlab编号从1开始，而再python中编号从0开始，所以所有编号都缩小1
        AreaNodeNumber = tempArea[0] - 1
        # print("AreaNodeNumber =", AreaNodeNumber)
        # np,show()
        AreaNodeCoordinateX = tempArea[1]
        AreaNodeCoordinateY = tempArea[2]
        # 将节点能量控制在50J以内
        AreaNodeEnergy = tempArea[3] * 0.01 * E_max
        # begin 锚点选择和优化
        # 初始选择锚点
        GridNode, AreaCellLabel, AreaAnchorList, AreaAnchorCoordinate = AnchorInitSelect(AreaNodeNumber, NodeCoordinatesX,
                                                                             NodeCoordinatesY, MaxGridNumber)

        #print(GridNode, "\n", AreaCellLabel, "\n", AreaAnchorList, "\n", AreaAnchorCoordinate)
        # 初始锚点展示，锚点在分区中心
        PoltInit(GridNode, AreaCellLabel, AreaAnchorList, AreaAnchorCoordinate, NodeCoordinatesX, NodeCoordinatesY)

        # 用于保存节点的能量
        # u_i_j = np.loadtxt("u_i_j.txt",float)
        if txtSaveFlag == 1:
            u_i_j = np.zeros([1, NodesNumber], dtype=float)
            for i in range(len(AreaCellLabel)):
                # 每个区域c[x][y]的序号为:x + MaxGridNumber*y
                # 知道序号后,反推x和y的值
                # 需要向下取整,得到当前列数
                # print("CellLabel[i] =", CellLabel[i])
                y = math.floor(AreaCellLabel[i] / MaxGridNumber)
                # print("y =", y)
                x = AreaCellLabel[i] - MaxGridNumber * y
                # print("x =", x)
                # 计算中心节点的坐标,该坐标即为锚点坐标
                # 当前锚点坐标（X,Y）
                X = (x * D) + (D / 2.0)
                Y = (y * D) + (D / 2.0)
                # print("X =", X)
                # print("Y =", Y)
                # 判断是否只存在一个传感器节点
                temp = GridNode[0][AreaCellLabel[i]]
                # print("temp =", temp)
                # if len(GridNode[0][CellLabel[i]]) != 1:
                for j in range(len(GridNode[0][AreaCellLabel[i]])):
                    # print("temp[j] =", temp[j])
                    # print("NodeCoordinatesX[temp[j]] =", NodeCoordinatesX[temp[j]])
                    # print("NodeCoordinatesY[temp[j]] =", NodeCoordinatesY[temp[j]])
                    tempdistance = math.sqrt(
                        math.pow((NodeCoordinatesX[temp[j]] - X), 2) + math.pow((NodeCoordinatesY[temp[j]] - Y), 2))
                    # print("tempdistance =", tempdistance)
                    u_i_j[0][temp[j]] = U_i_j(tempdistance)
                # else:
                # u_i_j[0][temp[0]] = U
            np.savetxt("u_i_j.txt", u_i_j)
        # print("u_i_j =", u_i_j)
        # 只有部分节点，所以所得结果也着用部分节点的

        #到此，第一步已经结束#
        # 节点的耗电率
        # 导入文件
        dataFileSense = 'data_sense_update.mat'
        Sensedata = scio.loadmat(dataFileSense)
        # print("Sensedata =", Sensedata['data_sense'])
        data_sense = Sensedata['data_sense']
        # data_sense_list = []
        # for data_sense_i in range(len(data_sense[0])):
        temp_data_sense = data_sense[0]
        # 第一个分区节点的数据感知率情况
        # print("temp_data_sense =", temp_data_sense[ID_i][0])
        AreaDataSense = temp_data_sense[ID_i][0]
            # data_sensevalue = data_sense[0][data_sense_i][0]
            # for data_sense_j in range(len(data_sensevalue)):
            #    data_sense_list.append(data_sensevalue[data_sense_j])
        # print("len(data_sense_list) =", len(data_sense_list))
        # print("data_sense_list =", data_sense_list, len(data_sense_list))

        # 节点剩余能量
        E_Sensor = np.zeros([1, NodesNumber], dtype=float)
        # 节点数据生成率
        R_i_Sensor = np.zeros([1, NodesNumber], dtype=float)
        # 节点能量消耗率
        p_i_Sensor = np.zeros([1, NodesNumber], dtype=float)
        E_s = 558e-8  # 感知单位bit数据能耗，J / bit
        # E_s = 0.018
        # print("E_s =", E_s)
        E_t = E_s  # % 传输电路能耗，J / bit
        E_r = E_s  # % 接收电路能耗，J / bit
        # 只记录当前分区节点的信息
        for i in range(len(AreaNodeNumber)):
            # 确定ID
            # print("i =", i)
            # print("AreaNodeIDList[i] =", AreaNodeIDList[i])
            # 节点剩余能量
            E_Sensor[0][AreaNodeNumber[i]] = AreaNodeEnergy[i]
            # 节点数据生成率
            # print("i =", i)
            # AreaDataSense[i]
            R_i_Sensor[0, :] = 10
            # 一方面是感知数据，另一方面是传输数据到MD，所以有两倍的功耗，从论文中得知，每秒需要消耗的能量
            p_i_Sensor[0, :] = 10 * E_s * 2
            # p_i_Sensor[0][AreaNodeIDList[i] - 1] = 0.01
        # print("E_Sensor =", E_Sensor)
        # print("R_i_Sensor =", R_i_Sensor)
        # print("p_i_Sensor =", p_i_Sensor)
        E_Sensor_Init_Sum = sum(E_Sensor[0])
        # print("初始节点能量E_Sensor_Init_Sum =", E_Sensor_Init_Sum)

        # 每个节点收集的数据
        R_Data = np.zeros([1, NodesNumber], dtype=float)

        AnchorCoordinate = APE_Strategy(GridNode, AreaCellLabel, NodeCoordinatesX, NodeCoordinatesY, p_i_Sensor[0],
                                        R_i_Sensor[0], u_i_j, G, MaxGridNumber)
        if testFlag == 0:
            print("AnchorCoordinate =", AnchorCoordinate)
        PlotOptimal(GridNode, AreaCellLabel, AnchorCoordinate, NodeCoordinatesX, NodeCoordinatesY)
        print("锚点优化结束！！！")

        # end 锚点选择和优化

        # begin 路径优化
        E_Time = np.zeros([1, NodesNumber], dtype=float)
        # 记录节点从充电结束后后，又运行了多久
        E_Time[0, :] = 0.0
        # 节点最小能量
        # E_min = 540.0

        # 小车速度-->保持与原文相同
        v = 3.0
        # 小车的能量传输效率
        eta = 0.3
        # 小车能耗，j/m
        P_T = 8.27

        Tour_L, AC_SSX, AC_SSY = PE_FWA(GridNode, AreaCellLabel, AnchorCoordinate)

        # Tour_L = [44, 35, 17, 23, 15, 7, 2, 1, 39, 22, 31, 36, 29, 24, 25, 12, 38, 18, 43, 21, 8, 28, 40, 32, 16, 0, 33, 27, 6, 10, 30, 5, 19, 20, 26, 42, 4, 37, 13,  3,  9, 14, 11, 41, 34, 44]
        # Tour_L = [44, 24, 25, 18, 19, 31, 42, 43, 37, 32, 26, 20, 13, 12,  6,  5,  4, 11, 17,  9,  3, 10,  2,  1,  0, 7,  8, 14, 15, 22, 21, 27, 28, 34, 33, 38, 39, 35, 40, 41, 36, 30, 29, 23, 16, 44]
        # print("len(Tour_L) =", len(Tour_L))
        # 利用烟花算法构建移动小车的最佳行驶路径
        if txtSaveFlag == 1:
            print("len(AnchorCoordinate[0]) =", len(AnchorCoordinate[0]))
            Tour_L, PEvalue, bestfitness = FW_main((len(AnchorCoordinate[0]) - 1), ID_i)

        if ID_i == 0:
            Tour_L = np.loadtxt("Tour_L.txt", dtype=int)
            D_L_List = np.loadtxt("Tour_length.txt", dtype=float)
            D_L = D_L_List[len(D_L_List) - 1]
        elif ID_i == 1:
            Tour_L = np.loadtxt("Tour_L1.txt", dtype=int)
            D_L_List = np.loadtxt("Tour_length1.txt", dtype=float)
            D_L = D_L_List[len(D_L_List) - 1]
        elif ID_i == 2:
            Tour_L = np.loadtxt("Tour_L2.txt", dtype=int)
            D_L_List = np.loadtxt("Tour_length2.txt", dtype=float)
            D_L = D_L_List[len(D_L_List) - 1]
        elif ID_i == 3:
            Tour_L = np.loadtxt("Tour_L3.txt", dtype=int)
            D_L_List = np.loadtxt("Tour_length3.txt", dtype=float)
            D_L = D_L_List[len(D_L_List) - 1]
        print("Tour_L =", Tour_L)
        print("len(Tour_L) =", len(Tour_L))
        # np,show()
        # 巡游路径的长度,之前保存的最佳距离值

        # MD迅游总时间
        # t_L = D_L/v
        # 把节点迅游路径描绘出来
        PlotTour(Tour_L, AC_SSX, AC_SSY, GridNode, AreaCellLabel, AnchorCoordinate, NodeCoordinatesX, NodeCoordinatesY)
        # 迅游路径就是Tour_L
        # end 路径优化结束
    np, show()
