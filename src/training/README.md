In this section we train a character based CNN to classify the scraped reviews into two categories (positive, negative)

To learn more about the model and the implementation, you can check the original <a href="https://github.com/ahmedbesbes/character-based-cnn">repo</a>


The model architecture comes from this paper: https://arxiv.org/pdf/1509.01626.pdf

![Network architecture](images/character_cnn.png)

There are two variants: a large and a small. You can switch between the two by changing the configuration file.

This architecture has 6 convolutional layers:

|Layer|Number of Kernels|Kernel size|Pool|
|-|-|-|-|
|1|256|7|3|
|2|256|7|3|
|3|256|3|N/A|
|4|256|3|N/A|
|5|256|3|N/A|
|6|256|3|3|

and 2 fully connected layers:

|Layer|Number of neurons|
|-|-|
|7|1024|
|8|1024|
|9|2|
