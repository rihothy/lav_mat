[English](README.md)

# lav_mat
这是一个基于boost和clBLAS的c++矩阵库。本库使用简单且使用了OpenCL进行加速。

## 背景
众所周知，深度学习需要用到大量的矩阵运算，而gpu的架构注定其适合做矩阵运算。目前gpu上的矩阵运算多用cuda，而我的显卡只支持OpenCL，不支持cuda，而基于OpenCL加速的矩阵库又有点难用（如果有好用的库请推荐给我），所以花了点时间，简单包装了boost（将OpenCL的c接口封装成c++接口）和clBLAS（提供基于OpenCL的矩阵运算），写了这个有点简陋但简单易用的gpu矩阵运算库，以支持我的深度学习之路。

## 安装
要使用这个库，首先要安装boost的[compute](https://github.com/boostorg/compute)模块和[clBLAS](https://github.com/CNugteren/CLBlast)库。然后在项目中包含本库的头文件就行了：
```c++
#include <lav_mat.h>
```

## 使用
本库用面向对象封装了一个Mat类，接口部分借鉴了numpy、matlab，不过大部分是我怎么高兴怎么写。  

#### 矩阵的初始化
矩阵初始化里的bool参数是指定矩阵初始化时，数据是否上传到显存。一般建议设为true，因为早晚要上去的。
```c++
#include <lav_mat.h>

using namespace lav;

int main(int argc, char* argv[])
{
    /*
    [[1 2 3]
     [4 5 6]]
    */
    Mat a(2, 3, { 1, 2, 3, 4, 5, 6 }, true);
    
    /*
    [[1 2 3 4 5 6]]
    */
    Mat b({ 1, 2, 3, 4, 5, 6 }, true);
    
    //生成2行3列矩阵，元素在-1到1之间均匀分布的矩阵
    Mat c = randu(2, 3, -1, 1, true)
    
    //生成2行3列矩阵，元素呈均值为4，方差为10的正态分布
    Mat d = randn(2, 3, 4, 10, true)
    
    //生成10阶单位矩阵
    Mat e = Eyes(10, true);
    
    //生成2行3列矩阵，元素全为0
    Mat f = Zeros(2, 3, true);
    
    //生成2行3列矩阵，元素全为1
    Mat g = Ones(2, 3, true);
    
    return 0;
}
```

#### 简单矩阵运算
矩阵间的运算支持广播（当然了，矩阵乘法不行）
```c++
#include <lav_mat.h>
#include <iostream>

using namespace lav;

int main(int argc, char* argv[])
{
    Mat a(2, 3, { 1, 2, 3, 4, 5, 6 });
    Mat b(2, 3, { 6, 5, 4, 3, 2, 1 });
    Mat c(3, 2, { 9, 8, 7, 6, 5, 4 });
    Mat d(2, 1, { 0.5, 0.3 });
    
    /*
    [[1.5 2.5 3.5]
     [4.3 5.3 6.3]]
    */
    std::cout << a + d << std::endl;
    
    /*
    [[2 4 6]
     [8 10 12]]
    */
    std::cout << 2 * a << std::endl;
    
    /*
    [[6 10 12]
     [23 20 6]]
    */
    std::cout << a * b << std::endl;
    
    /*
    [[38 32]
     [101 86]
    */
    std::cout << mul(a, c) << std::endl;
    
    /*
    [[1, 4, 9]
     [16, 25, 36]]
    */
    std::cout << pow(a, 2) << std::endl;
    
    //太简单了，不一一写了
    
    return 0;
}
```

#### 矩阵拼接、肢解操作
```c++
#include <lav_mat.h>
#include <iostream>

using namespace lav;

int main(int argc, char* argv[])
{
    Mat a(2, 3, { 1, 2, 3, 4, 5, 6 });
    Mat b(2, 3, { 0, 7, 6, 4, 9, 4 });
    
    /*
    [[1 2 3]
     [4 5 6]
     [0 7 6]
     [4 9 4]]
    */
    a.push_back(b);
    
    /*
    [[1 2 3]
     [4 5 6]
     [0 7 6]
     [4 9 4]
     [7 6 0]]
    */
    a.push_back({ 7, 6, 0 });
    
    /*
    [[4 5]
     [0 7]
    */
    std::cout << a(1, 3, 0, 2) << std::endl;
    
    /*
    9
    */
    std::cout << a(3, 1) << std::endl;
    
    /*
    [[4 9 4]]
    */
    std::cout << a.row(3) << std::endl;
    
    return 0;
}
```

#### 其他操作
```c++
#include <lav_mat.h>
#include <iostream>

using namespace lav;

int main(int argc, char* argv[])
{
    Mat a(2, 3, { 1, 2, 3, 4, 5, 6 });
    
    /*
    6
    */
    std::cout << a.max() << std::endl;
    
    /*
    [[4 5 6]]
    */
    std::cout << a.max(0) << std::endl;
    
    /*
    [[3]
     [6]]
    */
    std::cout << a.max(1) << std::endl;
    
    /*
    [[1 1 1]]
    */
    std::cout << a.max_loc(0) << std::endl;
    
    return 0;
}
```
