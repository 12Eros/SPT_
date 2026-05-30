#### 如需训练,修改以下文件中的路径 

1.\experiments\spt\unimod1k.yaml:预训练权重路径

2.lib\train\admin\local.py:最后两行，训练集地址与文本描述地址

#### 如需测试,修改以下文件中的路径

1.lib\test\parameter\spt.py:测试使用的权重

2.lib\test\evaluation\local.py:最后一行,测试集路径

##### 训练命令：

```bash
python -m lib.train.run_training
```

##### 测试命令

```bash
python -m tracking.test
```

##### 结果分析

```bash
python results_analysis/analyze.py   "test_dataset/path" "analysis_result_path"
```

##### 获取数据集：

https://pan.baidu.com/s/1R3NG_-FKv6Ztx_aBwQJ4-w?pwd=ctnk

https://drive.google.com/drive/folders/1Z2PnWEgdZG0KVI2MX5chWddNlbuuEug3?usp=share_link

##### 获取预训练权重:

[BERT pretrained weight](https://drive.google.com/drive/folders/1Fi-4TSaIP4B_TPi2Jme2sxZRdH9l5NPN?usp=share_link)

[Stark-s model](https://drive.google.com/drive/folders/142sMjoT5wT6CuRiFT5LLejgr7VLKmaC4)



