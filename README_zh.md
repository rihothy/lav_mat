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

#### 4维卷积
这里的4维卷积是指用来卷积的矩阵表达的是一个4维数组（虽然矩阵还是2维的），而卷积运算依然是2D的卷积。  
矩阵（参数为f）表示的数据如下  
![conv4d_f](https://github.com/rihothy/lav_mat/blob/master/images/conv4d_f.png)
filter矩阵（参数为g）表示的数据如下  
![conv4d_g](https://github.com/rihothy/lav_mat/blob/master/images/conv4d_g.png)
函数原型
```c++
Mat conv4d(Mat& f, Mat& g, std::vector<size_t> size, const size_t& stride, const std::string padding);
```
其中size为5维向量，值分别为w(width),h(height),channel,f(filter size),batch size，padding的值只能为"valid"或"same"。

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

## 性能分析
平台：windows10 x64  
CPU：Ryzen 2600  
GPU：Rx590  

其中gpu计算由本库支持，cpu计算由numpy支持。gpu计算时间不包括数据从内存转移到显存、OpenCL初始化、kernel编译的时间。

#### 矩阵乘法性能（底层计算由clBLAS提供）
矩阵阶数|gpu计算时间/ms|cpu计算时间/ms
:-:|:-:|:-:
16|0.125|0
32|0.125|0
64|0.125|0
128|0.125|0.125
256|0.1875|0.5
512|0.625|3.5
1024|2.1875|26
2048|9.875|176
4096|61.1875|1207
8192|389.25|10227
16384|2778.06|+∞

## 开源协议
Apache license 2.0
