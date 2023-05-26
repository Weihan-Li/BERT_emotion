# BERT实践：进行一般文本的情绪分类

BERT模型以及训练好的用于训练的h5模型上传在百度网盘：[BaiduDisk][https://pan.baidu.com/s/1wr7NVmHOeyHD6reChVKSFg]，提取码: 1234

目前情绪分类的准确率为68%左右

## 1.安装所需依赖

```
pip install transformers

pip install scikit-learn

pip install tensorflow==2.4.0

pip install panda==1.4.4

pip install numpy==1.19.5
```

tensorflow2.4.0对应CUDA版本为11.0，安装参考[windows 10 安装cuda11.0 cuDNN](https://blog.csdn.net/u011788214/article/details/117124772)

## 2.使用方法

安装好依赖后进入到文件目录，将解压好的h5模型和BERT模型文件夹放在文件目录下

在命令行中输入：

```
python BERT_predict.py
```

即可在命令行中实现输入文本并对情绪进行分类

输出结果为一个列表，列表中的六个值分别与情绪的对应如下表所示：

| 1    | 2    | 3    | 4      | 5    | 6    |
| ---- | ---- | ---- | ------ | ---- | ---- |
| 生气 | 害怕 | 积极 | 无情绪 | 悲伤 | 惊奇 |

在命令行中输入：

```
python BERT_trans.py
```

可以利用你的数据集对预训练模型（如BERT以及BERT的各种改进大模型）进行fine-tune

一般模型占用显存会比较严重，建议在显存大于10G的云服务器下进行训练

如果数据集并不是json格式或需要更改模型，请先对BERT_trans进行修改