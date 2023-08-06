# DORM-CSC
The repository for ACL 2023 paper: Disentangled Phonetic Representation for Chinese Spelling Correction.



## Data

### Download
Data is downloaded from ACL 2021 paper [ReaLiSe](https://github.com/DaDaMrX/ReaLiSe)

### Data Process
The data processing code is `data_process/data_processor.py`.

```sh
python data_process/data_processor.py
```


## Training and Evaluation

The code of Dorm is ``dorm.py``, and the code of training is ``dorm_finetune.py``.

The scripts of training and evaluation are ``train.sh`` and ``test.sh``, respectively.


# Acknowledgements
Our code was modified and developed based on [ReaLiSe](https://github.com/DaDaMrX/ReaLiSe), and we would like to express our gratitude to their team.

