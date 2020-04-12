# retina-unet
该项目的实现参考了[orobix/retina-unet](https://github.com/orobix/retina-unet), 本人仅对其部分代码进行了重构. 重构的目的是使项目可以基于tensorflow2运行, 并且增加代码的可读性.

## datasets

该项目的数据集有两个: DRIVE, CHASEDB, 以下是数据集的参数和样例.


参数\名称|DRIVE|CHASEDB
:-:|:--:|:-:
shape|(584, 565)|(960, 999)
train num|20|20
test num|20|8

下面是数据集的一个样本, 三幅图像从左到右依次是:

* 原始图像(数据集中为RGB图像, 为了展示, 此处转为了灰度图像).
* 标注图像, 白色(255)为血管标注, 黑色(0)为背景.
* 掩码图像, 用于区分眼球部分和非眼球部分.

![dataset_sample](./logs/dataset_sample.png)

DRIVE和CHASEDB两个数据集的下载地址如下:

* [DRIVE](https://pan.baidu.com/s/1M9k07LKul2c8gZBUzJ-TzA), 提取码: w2cf
* [CHAEDB](https://pan.baidu.com/s/1ZigFfnciLkQBd5AgMFWldg), 提取码: 6tac

## prepare datasets

在运行该项目之前, 首先需要准备数据集, 该过程如下:

1. 下载数据集.

2. 在项目的根目录下创建文件夹datasets.

3. 将下载所得的数据集解压到datasets中, 最终文件夹结构应该与下面一致:

   retina_unet/datasets/

   |-- CHASEDB

   ​    |-- test

   ​    |-- training

   |-- DRIVE

   ​    |-- test

   ​    |-- training

4. 运行rewrite_datasets.py.

以上过程会读取数据集, 然后将数据按类别分别写入*.hdf5文件, 这样做是为了使之后读取数据更简单.

## train model

## evaluation