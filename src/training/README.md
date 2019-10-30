In this section we train a character based CNN to classify the scraped reviews into two categories (positive, negative)

To learn more about the model and the implementation, you can check the original <a href="https://github.com/ahmedbesbes/character-based-cnn">repo</a>


The model architecture comes from this paper: https://arxiv.org/pdf/1509.01626.pdf

![Network architecture](plots/character_cnn.png)

There are two variants: a large and a small. You can switch between the two by changing the configuration file.

This architecture has 6 convolutional layers:

|Layer|Large Feature|Small Feature|Kernel|Pool|
|-|-|-|-|-|
|1|1024|256|7|3|
|2|1024|256|7|3|
|3|1024|256|3|N/A|
|4|1024|256|3|N/A|
|5|1024|256|3|N/A|
|6|1024|256|3|3|

and 2 fully connected layers:

|Layer|Output Units Large|Output Units Small|
|-|-|-|
|7|2048|1024|
|8|2048|1024|
|9|Depends on the problem|Depends on the problem|
