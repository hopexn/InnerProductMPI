# InnerProduct by MPI
#### 1. 使用方法

- 编译： mpicc InnerProduct InnerProduct.c
- 运行： mpirun -np <节点数> InnerProduct <向量长度> <向量个数>

#### 2. 说明
程序运行后，会根据用户输入生成一个N x M的矩阵(M为向量长度 N为向量个数), 然后随机生成一个矩阵
对这个矩阵进行串行和并行的内积运算，校验结果，若结果相同则输出Correct，错误显示Incorrect.

程序将N个向量分发到p个节点中，每个节点负责N/p个向量与其他向量的内积
结果为N x N的矩阵

例如：
现有一个向量矩阵M，向量个数为9, 使用3个节点对齐运算，每个节点计算3个向量与其他向量的内积
该节点的结果为(N/p) x N大小的矩阵，对应最终结果矩阵的N/p行

第1个节点负责计算1至3条向量的内积，首先计算这三条向量自己的内积, 也就是
第1至3行与第1至3行矩阵的内积

然后该节点将自己这一行的数据发送给下一个节点，并接受上一个节点发送的数据
对于第一个节点，将1至3行数据发送给第2个节点， 接受来自第3个节点的数据(7至9行)
然后计算1至3行向量与7至9行的内积

接下来重复上面的操作，向节点2发送数据，从节点3获得数据
节点1从节点3或得4~6行数据，节点2发送之前接受的数据（7至9行）
计算1至3行与4至6行向量内积

经过两轮消息传递，第一个节点计算得到1至3行与1至9行向量的内积，其他节点也运算完毕

然后将数据发送回主节点进行归并

最终的结果为9 x 9的内积矩阵

