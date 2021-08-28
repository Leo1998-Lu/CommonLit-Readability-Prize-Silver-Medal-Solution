## [CommonLit-Readability-Prize-Silver-Medal-Solution](https://www.kaggle.com/c/commonlitreadabilityprize/overview)
2021 Kaggle Featured Code Competition:CommonLit Readability Prize(Silver Medal Solution, Final ***Rank 96/3633*** teams

![image](https://user-images.githubusercontent.com/57436423/131209923-1340db4f-38b0-4ae4-86b3-df963d05fb5e.png)

- ****比赛任务****: 构建算法来评估 3-12 年级课堂使用的阅读文本段落的复杂性，用来评估文本的可读性，是否通俗易懂。
- ****评估指标****: RMSE
- ****比赛数据****：
  - id: 每条文本的唯一ID 
  - url_legal：数据来源，测试集中为空 
  - license ：数据许可协议，测试集中为空 
  - excerpt ：需要预测的测试集文本 
  - target ：可读性分数，目标值 
  - standard_error ：衡量每个摘录的多个评分者之间的分数分布。不包括测试数据

## **Solution**: RoBERTa Large/Base + Attention/Mean Head
**ITPT====>>Finetune====>>{RoBERTa Large/Base + Attention/Mean Head}====>>Inference**

•	根据target范围进行kfod数据划分( 1.train-val-split)  
•	基于比赛任务给定的训练集语料进行继续预训练：MLM任务   
•	对于预训练模型输出拼接其他网络层进行微调，主要用到的池化层有AttentionHead,MeanPooling   
•	融合:根据公榜分数设置权重进行加权相加 

- ****ITPT：继续预训练****
![image](https://user-images.githubusercontent.com/57436423/131210766-3ae03877-17ad-4f14-a24f-adf627d1f4a3.png)   
Bert是在通用的语料上进行预训练的，如果要在特定领域应用文本分类，数据分布一定是有一些差距的。这时候可以考虑进行深度预训练。Within-task pre-training：Bert在训练语料上进行预训练。(2.clrp-pretrain)

- ****不同层的特征****    
BERT的每一层都捕获输入文本的不同特征。文本研究了来自不同层的特征的有效性, 然后微调模型并记录测试错误率的性能。(3.clrp-finetune-roberta-large)

- ****模型层间差分学习率****   
发现为下层分配较低的学习率对微调Roberta-Large 是有效的，比较合适的设置是 ξ=0.9 和 lr=3.0e-5，其中24代表Large模型encoder层数，如果使用base需要改成12。

### Final Rank: 96/3633
![image](https://user-images.githubusercontent.com/57436423/131210883-a75e689c-8156-4857-9dc9-7e9ffb007889.png)

![image](https://user-images.githubusercontent.com/57436423/131210900-32d05865-1574-4a79-9ac0-542c4aa3fe79.png)

再次成功带队SOLO摘银，算是学习NLP近半年的成果验收:)!
