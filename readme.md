# DORM-CSC
The repository for ACL 2023 paper: Disentangled Phonetic Representation for Chinese Spelling Correction.

We will release the code as soon as possible.


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


# Acknowledgements
Our code was modified and developed based on [ReaLiSe](https://github.com/DaDaMrX/ReaLiSe), and we would like to express our gratitude to their team.

