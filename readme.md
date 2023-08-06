## 环境
代码是基于 [REALISE](https://arxiv.org/abs/2105.12306) 修改的，运行环境设置可参考该文章的仓库。

通常需要设置环境变量，把 `/..your_path../dorm_project_copy` 加入 `PYTHONPATH` 的环境变量中


## 数据

### 数据来源
数据来源于 ACL 2021 paper REALISE 的[开源库](https://github.com/DaDaMrX/ReaLiSe)

### 数据处理
处理代码在 `data_process/data_processor.py`，将数据转换为带有拼音的数据。

```sh
python data_process/data_processor.py
```


## 训练与测试

模型代码以及运行代码为 ``dorm.py`` 以及 ``dorm_finetune.py``。

训练以及评估模型的脚本为 ``train.sh`` 和 ``test.sh``。运行即可（已测试，成功运行）。


## 模型

原始的中文 BERT 模型、预训练后的模型、SOTA DORM 模型存放在 ``models`` 目录下。


## 总结
总的来说，这个项目是基于 [REALISE](https://arxiv.org/abs/2105.12306) 修改的。

除了以上涉及的具体 python 和 bash 文件外，其余许多文件都是 REALISE 项目的。

所以可以先跑通 REALISE 项目，随后再研究这一项目。