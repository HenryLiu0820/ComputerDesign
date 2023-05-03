# ComputerDesign

# File structure
```
├── b站爬虫
│   ├── controller
│   │   ├── **/*.css
│   ├── views
│   ├── model
│   ├── index.js
├── code
│   ├── dataloader.py
│   ├── finetune.py
│   ├── prediction.py
│   ├── test.ipynb
|   ├── utils.py
├── data
├── data_analyze.ipynb
├── 文档
└── README.md
```

<!--Write command demonstration-->
# Run the training module
**Run the following command in the terminal in one line(add space between each arg), the content in the [bracket] is the customizable parameter to be filled in**

```bash
conda run -n base --no-capture-output --live-stream python [Directory of the finetune.py file] \
  --device [str: device to use, i.e. 'cuda:0'] \
  --if_local [bool: whether to load existing model from local, **note: the name of the existing model has to be the same with the model_name declared below**] \
  --model_name [str: name of the pre-trained model] \
  --epochs [int: num of epochs] \
  --batch_size [int: batch size] \
  --weight_decay [float: weight decay] \
  --drop_prob [float: dropout probability] \
```

# Run the prediction module
**Run the following command to predict the label using costomized dataset. Note: must save your dataset to the /data directory.**
```bash
conda run -n base --no-capture-output --live-stream python [Directory of the prediction.py file] \
  --model_path [str: path to the model, default: /prediction.py] \
  --dataset_name [str: name of dataset(must be stored in the /data directory)] \
  --device [str: device to use, i.e. 'cuda:0'] \
  --model_name [str: name of the tokenizer model] \
``` 
**Example**
```bash

conda run -n base --no-capture-output --live-stream python /Users/zhengyuan/Desktop/ComputerDesign/code/prediction.py --model_path /Users/zhengyuan/Desktop/ComputerDesign/code/model/finetuned_model.pt --dataset_name test --device cuda:0 --model_name bert-base-chinese
```
