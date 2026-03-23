# Data

This part preprocess the data into interleaved form. Currently the data are from:

- [EpiCoder](https://huggingface.co/datasets/microsoft/EpiCoder-func-380k)

Yaml tag for dataset are as below:
```
---
configs:
- config_name: EpiCoder
  data_files:
  - split: train
    path:
    - "epicoder/*.parquet"
---
```
Put the yaml tag at the top of the dataset `README.md` file for automatic loading.
