# 前言
主要使用pytorch 实现无监督Simsce和whitening，参考：

Simsce     : https://github.com/bojone/SimCSE
whitening : https://github.com/bojone/BERT-whitening

# 数据准备
由于实验的数据有保密性，这里暂不提供，可使用自己领域的任意数据，形式如train_simsce.xlsx和test_simsce.xlsx

# 训练
## Simsce训练和提取特征
```python SimBERT_train.py
python SimBERT_predict.py
```
之后会生成feature_train.npy 和 feature_test.npy 分别代表train和test在Simsce训练后提取的对应特征。

## whitening训练和提取特征
将whitening.py的133行的is_whitening=True
```
python whitening.py
```
之后会生成whitening_train.npy 和 whitening_test.npy 分别代表train和test在whitening训练后提取的对应特征。


## 原始预训练模型的特征
为了对比，我们可以直接用bert的原始向量来做做相似度
将whitening.py的133行的is_whitening=False
```
python whitening.py
```
之后会生成original_train.npy 和 original_test.npy 分别代表train和test在原始bert提取的对应特征。

# 计算相似度score
将get_score.py的25，26行分别替换为上面一种方式对应的train和test的npy即可得到对应方式的分数结果。
get_score.py的19，20行,分别代表看多少个测试数据和每个测试数据返回的最相似的TopK
```
python get_score.py
```
随后生成的结果为result.xlsx
