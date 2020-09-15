# Text_CNN
******************
#### 环境
* Pytorch
* gensim
* python常用包
***************
* 用法
* 1 通过fasta2word将fasta先进行分词
```
python fasta2word.py -fasta xxx.fasta
```
* 2 设置参数
```
# 必要参数
-train_data_path
/Users/huangqianfei/Desktop/6mA/data/880/trainword.txt
-train_pos
880
-train_neg
880
# 非必要参数
-test_data_path
xxx.fasta
-test_pos
800
-test_neg
800
```

******************************
![Text-CNN](https://github.com/huangqianfei0916/Text_CNN/blob/master/1.png)
