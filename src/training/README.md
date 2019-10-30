In this section we train a character based CNN to classify the scraped reviews into two categories (positive, negative)

To learn more about the model and the implementation, you can check the original <a href="https://github.com/ahmedbesbes/character-based-cnn">repo</a>


The model architecture comes from this paper: https://arxiv.org/pdf/1509.01626.pdf

![Network architecture](images/character_cnn.png)

We'll be using the small variant of this model that has:

- 6 convolutional layers:

|Layer|Number of Kernels|Kernel size|Pool|
|-|-|-|-|
|1|256|7|3|
|2|256|7|3|
|3|256|3|N/A|
|4|256|3|N/A|
|5|256|3|N/A|
|6|256|3|3|

- 2 fully connected layers:

|Layer|Number of neurons|
|-|-|
|7|1024|
|8|1024|
|9|2|

## Requirements

- PyTorch 0.4.1
- Tensorflow 2.0.0 (optional, useful for model monitoring)
- TensorboardX 1.8 (optional, useful for model monitoring) 

## Training

When the scraping is done and the data is downloaded to src/scraping/scrapy/comments_trustpilot.csv

```shell
cd src/training/

python train.py --data_path ../src/scraping/scrapy/comments_trustpilot.csv \
                --validation_split 0.1 \
                --label_column rating \
                --text_column comment \
                --group_labels binarize \ 
                --extra_characters "éàèùâêîôûçëïü" \
                --max_length 500 \
                --dropout_input 0 \
                --model_name trustpilot \
                --balance 1
```

To learn more about the training options please check the <a href="https://github.com/ahmedbesbes/character-based-cnn">original repo</a>.

## Model monitoring 

```shell
tensorboard --logdir=src/training/logs/ --port=6006
```

## Use the trained model

Once the training is done you'll find a bunch of model checkpoints saved into the src/training/models folder

Select the one of your choice, rename it to **model.pth** and copy it in the src/api/ml/checkpoints/ folder

The api scripts will be in charge of the inference based on this model
