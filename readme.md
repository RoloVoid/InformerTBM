# ReadMe

* 毕业设计支撑材料：基于Informer的盾构机滚刀磨损数据预测研究
* 这是一个基于盾构机（TBM）滚刀数据在informer源码的基础上进行的适配和修改，主要修改包括
  1. 适配盾构机数据
  2. 根据盾构机数据简单的时序特征修改embedding，简化原模型
  3. 学习率工具的调整
  4. 部分结构的简化，对于可用数据量较小的盾构机数据无需更大规模的编码解码层
* 在训练时根据hyper-param.yml.exp修改自己的参数文件并命名为hyper-param以正常训练
* 核心参考文献：
* Zhou, Haoyi, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.” *Proceedings of the AAAI Conference on Artificial Intelligence* 35, no. 12 (May 18, 2021): 11106–15. [https://doi.org/10.1609/aaai.v35i12.17325](https://doi.org/10.1609/aaai.v35i12.17325).
