# Chat Robot

---

## UPDATE 2020.4.20

---

第一次上传项目

## 项目概述

---

- 本项目为基于微信的对话机器人。
- 对话机器人使用了三种回复机制：数据库搜索（针对业务相关问题）、网络爬虫（针对专业性问题，比如返回百度百科的词条内容）和生成系统（针对闲聊）。
- 项目结构如下

![structure.jpg](https://github.com/kkb-Alex/job_hunting/blob/master/ChatRobot/structure.jpg)

## 依赖环境、库

---

Python3、gensim、sklearn、numpy、pandas、lxml、torch、transformers

## 项目结构

- autoreply:
  - autoresponse.py: 汇总三种回复机制
  - editDistance.py: 编辑距离函数
  - searchInDatabase.py: 数据库搜索机制
  - searchBySpider.py: 基于百度百科和百度知道的爬虫搜索机制
  - GenerateByGPT2.py: 基于GPT2生成回复机制
- model: wordembedding模型和GPT2模型
- data: QA数据库文件
- wechatreply.py: 启动微信服务（需要自己搭建隧道）

## 模型及数据集分享

---

[robot's data&model](https://pan.baidu.com/s/1AgxrqAEqjbUfc6a7CNHf8g)(提取码p3mc)

## 使用样例

---

sample1

```
Q: 海苔麻花怎么做？
A: 在网上买的，网上也可以做，这个是无印良品的懒人做的，不过我觉得还是网上买的好吃(From GPT2)
```

sample2

```
Q: 你好呀！
A: 在呢在呢！(From Database)
Q: 我想办理信用卡
A: 各家银行办理信用卡的基本要求都是一致的，要年满18周岁，有稳定工作收入，信用良好，当然，如果有一定的资产会更好，综合以上，如果你没有工作，最好是提供资产类证明尝试申请信用卡总结了以下银行渠道的办卡方式：第一、通过银行网点办卡，要带好身份证、资产证明，像房产、车产、银行的定期存单、理财等等，那么对下卡就是非常有帮助的。第二、现在也有银行可以通过网络渠道办卡（我用的是建行，他家网站办卡比网点办卡方便很多呢，主要他家网点够多，还款便捷，最近活动也挺丰富的，用着不错），你也不妨可以试试。(From 百度知道)
```

sample3

```
Q: 成龙
A: 成龙（Jackie Chan），1954年4月7日出生于香港，祖籍安徽芜湖，中国香港男演员、导演、动作指导、制作人、编剧、歌手。(From 百度百科)
```

sample4

```
Q: 在？
A: 在啊(From GPT2)
Q: 我好困
A: 那您赶紧睡觉吧！(From Database)
Q: 我想吃鸡腿
A: 哈哈，我也这么想，可以嘛(From Database)
Q: 信用卡办理
A: 您拨打信用卡热线4008200588办理即可(From Database)
Q: 可以微信扫码么？
A: 不方便啊，只能微博扫码(From GPT2)
Q: 我喜欢看金庸小说
A: 我也喜欢看，但是我看的是悲剧(From GPT2)
Q: 冠状病毒
A: 冠状病毒在系统分类上属套式病毒目（Nidovirales）冠状病毒科（Coronaviridae）冠状病毒属（Coronavirus）。冠状病毒属的病毒是具囊膜（envelope）、基因组为线性单股正链的RNA病毒，是自然界广泛存在的一大类病毒。(From 百度百科)
```

## Future Work

- 增加槽填充，以应对诸如询问天气、查询车票等日常任务需求
- 增加多轮对话能力
- 手动对数据库里的QA进行类别标注，这样可以对输入的问题先进行分类，然后分配给不同功能的回复机制。



