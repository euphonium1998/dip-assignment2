## 其实没有什么特别的方式
就是hog提取特征

然后sgd/svm跑一层就结束了

SIFT提的特征不好用

网上就这两个比较好的特征提取方法

输入图片的预处理就是做个延y轴翻转做了一次增强

## 预处理
对训练集的每张输入图片按比例随机裁剪10次，然后每张裁剪后的图片再进行沿y轴翻转。处理后的20张图片加入训练集中。