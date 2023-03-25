# ComputerDesign

# File structure
* README.md
* /code
  * /dataloader.py
  * /finetune.py
  * /prediction.py
  * /test.ipynb
  * /utils.py
* /data

<!--Write demonstration command-->
# Run the training module
**Run the following command in the terminal in one line(add space between each arg), the content in the [bracket] is the customizable parameter to be filled in**

```bash
conda run -n base --no-capture-output --live-stream python [Directory of the finetune.py file]
  --device [str: device to use, i.e. 'cuda:0']
  --if_local [bool: whether to load existing model from local, **note: the name of the existing model has to be the same with the model_name declared below**]
  --model_name [str: name of the pre-trained model]
  --epochs [int: num of epochs]
  --batch_size [int: batch size] 
  --weight_decay [float: weight decay]
  --drop_prob [float: dropout probability]


