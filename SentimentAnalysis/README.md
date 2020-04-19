# Sentiment Analysis

---

## UPDATE 2020.4.19

---

第一次上传项目

## 项目概述

---

- 本项目为AI Challenger 2018：细粒度用户评论情感分类。一共有6大类20小类的分类目标。
- 由于只获得了105000条训练数据，因此选取5000条作为测试集，100000条作为训练集
- 经过多次随机分配测试集和训练集，平均f1在0.72～0.73之间。
- 基本思路是将该问题看作20个aspect的情感多分类问题，用两层双向LSTM+Attention作为共享层，分类层为独占层（避免建立20个模型，导致训练过慢，存储成本过高）的模型来训练
- 模型如下：

![modelstructure](https://github.com/kkb-Alex/job_hunting/tree/master/SentimentAnalysis/modelstructure.jpg)

## 依赖环境、库

---

Python3、Numpy、Pandas、Jieba、tensorflow(2.0)

## 项目结构

---

- att.py: Attention层
- data: 存放数据的文件夹
- dataprocessing.py: 数据处理
- main.py: 启动数据处理及训练
- model: 存放模型的文件夹
- model.py: tensorflow模型
- predict.py: 预测用
- static: css文件夹
- templates: html文件夹
- train.py: 训练模型
- utils.py: 分词等函数
- web_main.py: 启动网页端服务

## 模型及数据集分享

---

[data&model](https://pan.baidu.com/s/1hD6p9yW_3K-UuAqSNoJBPg)(提取码yw21)

## 使用样例(predict.py)

---

sample1

```
Comment:
第三次参加大众点评网霸王餐的活动。这家店给人整体感觉一般。首先环境只能算中等，其次霸王餐提供的菜品也不是很多，当然商家为了避免参加霸王餐吃不饱的现象，给每桌都提供了至少六份主食，我们那桌都提供了两份年糕，第一次吃火锅会在桌上有这么多的主食了。整体来说这家火锅店没有什么特别有特色的，不过每份菜品分量还是比较足的，这点要肯定！至于价格，因为没有看菜单不了解，不过我看大众有这家店的团购代金券，相当于7折，应该价位不会很高的！最后还是要感谢商家提供霸王餐，祝生意兴隆，财源广进
Result:
location_traffic_convenience---------------- Not mention
location_distance_from_business_district---- Not mention
location_easy_to_find----------------------- Not mention
service_wait_time--------------------------- Not mention
service_waiters_attitude-------------------- Not mention
service_parking_convenience----------------- Not mention
service_serving_speed----------------------- Not mention
price_level--------------------------------- Normal
price_cost_effective------------------------ Not mention
price_discount------------------------------ Good
environment_decoration---------------------- Normal
environment_noise--------------------------- Normal
environment_space--------------------------- Normal
environment_cleaness------------------------ Normal
dish_portion-------------------------------- Good
dish_taste---------------------------------- Good
dish_look----------------------------------- Not mention
dish_recommendation------------------------- Not mention
others_overall_experience------------------- Good
others_willing_to_consume_again------------- Not mention
```

sample2

```
Comment:
特意搜了一下排面靠前都这家店，总体来说很失望。
一家三口点了五个菜都没吃饱，一份虾58元每只都很小，分量小的可怜，盘子但是挺大，问了服务员说"就是这么点"。
蛤蜊好多都是空都，肉都不知道掉哪里去了，再也不会光顾。
Result:
location_traffic_convenience---------------- Not mention
location_distance_from_business_district---- Not mention
location_easy_to_find----------------------- Not mention
service_wait_time--------------------------- Not mention
service_waiters_attitude-------------------- Bad
service_parking_convenience----------------- Not mention
service_serving_speed----------------------- Not mention
price_level--------------------------------- Bad
price_cost_effective------------------------ Not mention
price_discount------------------------------ Not mention
environment_decoration---------------------- Not mention
environment_noise--------------------------- Not mention
environment_space--------------------------- Not mention
environment_cleaness------------------------ Not mention
dish_portion-------------------------------- Bad
dish_taste---------------------------------- Not mention
dish_look----------------------------------- Not mention
dish_recommendation------------------------- Not mention
others_overall_experience------------------- Bad
others_willing_to_consume_again------------- Bad
```

## Web样例

---

![web_example.png](https://github.com/kkb-Alex/job_hunting/tree/master/SentimentAnalysis/web_example.png)