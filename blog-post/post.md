# End to End Machine Learning : From Data Collection to Deployment 🚀

## Introduction

In this post, we'll go through the necessary steps to build and deploy a machine learning application. This starts from data collection to deployment and the journey, as you'll see it, is exciting and fun 😀.

Before we begin, let's have a look at the app we'll be building:

<p align="center">
    <img src="./assets/app.gif">
</p>

As you can see, this web app allows a user to evaluate random brands by writing reviews. While writing, the user will see the sentiment score of his input updating in real-time along with a proposed rating from 1 to 5.

The user can then change the rating (in case the suggested rating does not reflect his views), and submit.

You can think of this as a crowd sourcing app of brand reviews with a sentiment analysis model that suggests ratings that the user can tweak and adapt afterwards.

To build this application we'll follow the following steps:

- Data collection
- Model training
- App development
- Containerization
- App deployment

All the code is available in github and organized in independant directories.

Let's get started! 💻

## Scraping the data from Trustpilot with Selenium and Scrapy ?

In order to train a sentiment classifier, we need data. We can sure download open source datasets for sentiment analysis tasks such as <a href="http://jmcauley.ucsd.edu/data/amazon/"> Amazon Polarity</a> or <a href="https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset">IMDB</a> movie reviews but for the purpose of this tutorial, **we'll build our own dataset**. We'll scrape customer reviews from Trustpilot. 

Trustpilot.com is a consumer review website founded in Denmark in 2007. It hosts reviews of businesses worldwide and nearly 1 million new reviews are posted each month.


<p align="center">
  <img src="./assets/truspilot.png" width="90%">
</p>


Trustpilot is an interesting source because each customer review is associated with a number of stars.

<p align="center">
  <img src="./assets/review_label.png" width="70%">
</p>

By leveraging this data, we are able to map each review to a sentiment class. 

In fact, reviews with:

- 1 and 2 stars are **bad reviews** ❌
- 3 stars are **average reviews** ⚠️
- 4 and 5 stars are **good reviews** ✅


In order to scrape customer reviews from trustpilot, we first have to understand the structure of the website. 

Trustpilot is organized by categories of businesses.

<p align="center">
  <img src="./assets/1-categories.png" width="80%">
</p>

Each category is divided into sub-categories.

<p align="center">
  <img src="./assets/2-subcategories.png" width="80%">
</p>

Each sub-category is divided into companies.

<p align="center">
  <img src="./assets/3-companies.png" width="80%">
</p>

And then each company has its own set of reviews, usually spread over many pages.

<p align="center">
  <img src="./assets/4-reviews.png" width="80%">
</p>


As you see, this is a top down tree structure. In order to scrape it efficiently we'll use **Scrapy** framework, but before going that far we need a to use Selenium first to fetch the company urls (see previous screenshot), then feed those to Scrapy.

We need to use Selenium because the content of the website that renders those urls is dynamic (but the rest is not) and cannot be accessed directly from the page source like Scrapy does. Selenium simulates a browser that clicks on each category, narrows down to each sub-category and finally goes through all the companies one by one and fetches their urls. When it's done, the script saves these urls to a csv file and the Scrapy part can be launched.

### Scrape company urls with Selenium : step 1 

Let's see how to launch Selenium to fetch the company urls.

We'll first import Selenium dependencies along with other utility packages.

```python
import json
import time

from bs4 import BeautifulSoup
import requests
import pandas as pd

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from tqdm import tqdm_notebook

base_url = "https://trustpilot.com"


def get_soup(url):
    return BeautifulSoup(requests.get(url).content, 'lxml')
    
```

We start by fetching the sub-category URLs nested inside each category.

If you open up your browser and inspect the source code, you'll find out 22 category blocks (on the right) located in `div` objects that have a `class` attribute equal to  `category-object`

<p align="center">
    <img src="./assets/5-category_block.png" width="80%">
</p>

Each category has its own set of sub-categories. Those are located in `div` objects that have `class` attributes equal to `child-category`.
<p align="center">
    <img src="./assets/6-nested_urls.png" width="80%">
</p>


Let's first loop over categories and for each one of them collect the URLs of the sub-categories. This can be achieved using Beautifulsoup and requests.

```python
data = {}

soup = get_soup(base_url + '/categories')
for category in soup.findAll('div', {'class': 'category-object'}):
    name = category.find('h3', {'class': 'sub-category__header'}).text
    name = name.strip()
    data[name] = {}  
    sub_categories = category.find('div', {'class': 'sub-category-list'})
    for sub_category in sub_categories.findAll('div', {'class': 'child-category'}):
        sub_category_name = sub_category.find('a', {'class': 'sub-category-item'}).text 
        sub_category_uri = sub_category.find('a', {'class': 'sub-category-item'})['href'] 
        data[name][sub_category_name] = sub_category_uri
```

Now comes the selenium part: we'll need to loop over the companies of each sub-category and fetch their URL. 

We first define a function to fetch company urls referenced in a given subcategory:

```python
def extract_company_urls_form_page():
    a_list = driver.find_elements_by_xpath('//a[@class="category-business-card card"]')
    urls = [a.get_attribute('href') for a in a_list]
    dedup_urls = list(set(urls))
    return dedup_urls
```

and another function to check if there is a next page:

```python
def go_next_page():
    try:
        button = driver.find_element_by_xpath('//a[@class="button button--primary next-page"]')
        return True, button
    except NoSuchElementException:
        return False, None
```

Now we initialize Selenium with a headless Chromedriver. 

PS: You'll have to donwload Chromedrive from this <a href="https://chromedriver.chromium.org/">link</a> 


```python
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('start-maximized')
options.add_argument('disable-infobars')
options.add_argument("--disable-extensions")

prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome('./driver/chromedriver', options=options)

timeout = 3
```

and launch the scraping. This approximatively takes 50 minutes with good internet connexion:

```python
company_urls = {}
for category in tqdm_notebook(data):
    for sub_category in tqdm_notebook(data[category], leave=False):
        company_urls[sub_category] = []

        url = base_url + data[category][sub_category] + "?numberofreviews=0&timeperiod=0&status=all"
        driver.get(url)
        try: 
            element_present = EC.presence_of_element_located(
                (By.CLASS_NAME, 'category-business-card card'))
            
            WebDriverWait(driver, timeout).until(element_present)
        except:
            pass
    
        next_page = True
        c = 1
        while next_page:
            extracted_company_urls = extract_company_urls_form_page()
            company_urls[sub_category] += extracted_company_urls
            next_page, button = go_next_page()
            
            if next_page:
                c += 1
                next_url = base_url + data[category][sub_category] + "?numberofreviews=0&timeperiod=0&status=all" + f'&page={c}'
                driver.get(next_url)
                try: 
                    element_present = EC.presence_of_element_located(
                        (By.CLASS_NAME, 'category-business-card card'))
                    
                    WebDriverWait(driver, timeout).until(element_present)
                except:
                    pass
```

Once the scraping is over, we save the company urls to a csv file.

```python
consolidated_data = []

for category in data:
    for sub_category in data[category]:
        for url in company_urls[sub_category]:
            consolidated_data.append((category, sub_category, url))

df_consolidated_data = pd.DataFrame(consolidated_data, columns=['category', 'sub_category', 'company_url'])

df_consolidated_data.to_csv('./exports/consolidate_company_urls.csv', index=False)
```

And here's what the data looks like:

<p align="center">
    <img src="./assets/url_companies.png" width="80%">
</p>

### Scrape customer reviews with Scrapy : step 2

Ok, now we're ready to scrape the data we need with Scrapy.

First you need to make sure Scrapy is installed. Otherwise, you can install it using
- conda: `conda install -c conda-forge scrapy` 

or 

- pip: `pip install scrapy`

Then, you'll need to start a scrapy project:

```bash
cd src/scraping/scrapy
scrapy startproject trustpilot
```

This last command creates the structure of a Scrapy project. Here's what it looks like:

```
scrapy/
    scrapy.cfg            # deploy configuration file

    trustpilot/             # project's Python module, you'll import your code from here
        __init__.py

        items.py          # project items definition file

        middlewares.py    # project middlewares file

        pipelines.py      # project pipelines file

        settings.py       # project settings file

        spiders/          # a directory where you'll later put your spiders
            __init__.py
```

Using Scrapy for the first time can be overwhelming, so to learn more about it you can visit the official <a href="http://doc.scrapy.org/en/latest/intro/tutorial.html">tutorials</a>

To build our scraper, we'll have to create a spider inside the `spiders` folder. We'll call it `scraper.py`.

What the scraper will do is the following:

- It starts from a company url
- It goes through each customer review and yields a dictionary of data cotaining the following items

    - comment: the text review
    - rating: the number of stars (1 to 5)
    - url_website: the company url on trustpilot 
    - company_name: the company name being reviewed
    - company_website: the website of the company being reviewed
    - company_logo: the url of logo of the company being reviewed 
- It moves to the next page if any

Here's the full script.

To fully understand it, you should inspect the source code. It's really easy to get. 

In any case, if you have a question don't hesitate to post it in the comment section ⬇

```python
import re
import pandas as pd
import scrapy

class Pages(scrapy.Spider):
    name = "trustpilot"

    company_data = pd.read_csv('../selenium/exports/consolidate_company_urls.csv')
    start_urls = company_data['company_url'].unique().tolist()

    def parse(self, response):
        company_logo = response.xpath('//img[@class="business-unit-profile-summary__image"]/@src').extract_first()
        company_website = response.xpath("//a[@class='badge-card__section badge-card__section--hoverable']/@href").extract_first()
        company_name = response.xpath("//span[@class='multi-size-header__big']/text()").extract_first()
        comments = response.xpath("//p[@class='review-content__text']")
        comments = [comment.xpath('.//text()').extract() for comment in comments]
        comments = [[c.strip() for c in comment_list] for comment_list in comments]
        comments = [' '.join(comment_list) for comment_list in comments]

        ratings = response.xpath("//div[@class='star-rating star-rating--medium']//img/@alt").extract()
        ratings = [int(re.match('\d+', rating).group(0)) for rating in ratings]

        for comment, rating in zip(comments, ratings):
            yield {
                'comment': comment,
                'rating': rating,
                'url_website' : response.url,
                'company_name': company_name,
                'company_website': company_website,
                'company_logo': company_logo
            }

        next_page = response.css('a[data-page-number=next-page] ::attr(href)').extract_first()
        if next_page is not None:
            request = response.follow(next_page, callback=self.parse)
            yield request

```

Before launching the scraper, you have to change a couple of things in the settings.py:

Here are the changes we made:

```python
# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 32

#Export to csv
FEED_FORMAT = "csv"
FEED_URI = "comments_trustpilot_en.csv"
```
This indicates to the scraper to ignore robots.txt, to use 32 concurrent requests and to export the data into a csv format under the filename: `comments_trustpilot_en.csv`

Now time to launch the scraper: 

```bash 
cd src/scraping/scrapy
scrapy crawl trustpilot
```

We'll let it run for a little bit of time.

Note that we can interrupt it at any moment since it saves the data on the fly.

❗ This script is meant for educational purposes only: scrape responsively. ❗

## Training a sentiment classifer usig PyTorch 🤖

*The code and the model we'll be using here are inspired from this github <a href="https://github.com/ahmedbesbes/character-based-cnn">repo</a> so go check it for additional information.*

Now that the data is collected, we're ready to train a sentiment classifier to predict the labels we defined earlier.

There are a wide range of possible models to use. The one we'll be training is a character based convolutional neural network. It's based on this <a href="https://arxiv.org/pdf/1509.01626.pdf">paper</a> and it has proven to be really good on text classification.

The question you'd be asking up-front though is the following : how would you use CNNs for text classification ? Aren't these architectures designed for image data ?

Well, the truth is, CNN are way more versatile and their application can extend the scope of image classification. In fact, they are also able to capture sequential information that is inherent to text data. The only only trick here is to efficiently represent the input text.

To see how this is done, imagine the following tweet:

<p align="center">
    <img src="./assets/tweet.png" width="60%">
</p>

Assuming an alphabet of size 70 containing the english letters and the special characters and an arbitrary maximum length of 140, one possible representation of this sentence is a (70, 140) matrix where each column is a one hot vector indiciating the position of a given character in the alphabet and 140 being the maximum length of tweets. This porcess is called **quantization**.

Note that if a sentence is too long, the representation truncates up to the first 140 characters. On the other hand, if the sentence is too short 0 column vectors are padded until the (70, 140) shape is reached.

So what to do now with this representation?

<p align="center">
    <img src="./assets/tweet_matrix.png" width="60%">
</p>

**Feed it to a CNN for classification, obviously 😁**

But there's a small trick though. Convolutions are usually performed using 2D-shaped kernels, because these structures capture the 2D spatial information lying in the pixels. 
Text is however not suited to this type of convolutions beacuse letters follow each other sequentially, in one dimension only, to form a meaning. To capture this 1-dimensional denpendency, we'll use **1D convolutions**.

**So how does a 1-D convolution work?**

Unlike 2D-convolutions that make a 2D kernel slide horizontally and vertically over the pixels, 1D-convolutions use 1D kernels that slide horizontally only over the columns (i.e. the characters) to capture the dependency between characters and their componsitions.

The diagram below shows the architecture we'll be using: 

<p align="center">
    <img src="./assets/character_cnn_architecture.png" width="80%">
</p>

It has 6 convolutional layers:

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
|9|3|

On the raw data, convolutions with a kernel of size 7 are applied. Then the output of this layer is fed to a second convolution layer with a kernel of size 7 as well, etc.

After the last convolution layer, the output is flattened and passed through two successive fully connected layers before a classification layers.

Character CNN are interesting for various reasons since they have nice properties:

- They are quite powerful in text classification (see paper's benchmark) even though they don't have any notion of semantics
- You don't need to apply any text preprocessing (tokenization, lemmatization, stemming ...) while using them
- They handle misspelled words and OOV (out-of-vocabulary) tokens
- They are faster to train compared to recurrent neural networks
- They are lightweight since they don't require storing a large word embedding matrix. Hence, you can deploy them in production easily


That's all about the theory now, if you're still interested you can check this video tutorial made by me to fully understand character level CNNs.

### How to train the model using PyTorch

In order to train a character level cnn, you'll find all the files you need under the `src/training/` folder.

Here's the structure of the code inside this folder:
- `train.py`: used for training a model
- `predict.py`: used for the testing and inference
- src: a folder that contains:
    - `model.py`: the actual CNN model (model initialization and forward method
    - `dataloader.py`: the script responsible of passing the data to the training after processing 
    - `utils.py`: a set of utility functions for text preprocessing (url/hashtag/user_mention removal)


To train a classifer our classifier, run the following commands:

```bash

cd src/training/

python train.py --data_path ../src/scraping/scrapy/comments_trustpilot_en.csv \
                --validation_split 0.1 \
                --label_column rating \
                --text_column comment \
                --group_labels 1 \ 
                --max_length 500 \
                --dropout_input 0 \
                --model_name trustpilot \
                --balance 1
```

To learn more about the training arguments and options, please check out the original <a href="https://github.com/ahmedbesbes/character-based-cnn">repo</a>.

## Building an interactive web app 📲 with Dash, Flask and PostgeSQL

Now that we have trained the sentiment classifier, let's build our application so that end-users can interact with the model and evaluate new brands.

Here is a schema of our app architecture :


<p align="center">
    <img src="./assets/7-application_schema.png" width="80%">
</p>

As you can see, there are four building blocks for our app:
- A visualization application built using [Dash](https://dash.plot.ly/).
- A [Flask](https://flask.palletsprojects.com/en/1.1.x/) REST API.
- A [PostgreSQL](https://www.postgresql.org/) database
- Our trained Machine Learning model.

The Dash app will make http requests to the Flask API, wich will in turn interact with either the PostgreSQL database or the ML model, in order to respond.

If you are already familiar with Dash, you know that it is built on top of Flask. So we could basically get rid of the API and get everything done from within the Dash app.

We chose not to for a very simple reason: it makes the logic and the visualization parts independant. Indeed, because we have a separated API, we could with very little effort replace the Dash app with any other frontend technology, or add a mobile or desktop app.


Now, let's have a closer look at how those blocks are built.

### PostgreSQL Database

Nothnig fancy or original for the database part. We chose to use one of the most widely used relational database, PostgreSQL.

To run a PostgreSQL database for local development, you can either download PostgreSQL from the [official website](https://www.postgresql.org/download/) or, more simply, launch a postgres container using [Docker](https://www.docker.com/):

```
docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres -p 5432:5432 -d postgres
```

### Flask API

The RESTful API is the most important part of our app. It is responsible for the interractions with both the machine learning model and the database.

Let's have a look at the routes that we need for our api:


- Sentiment Classification : `POST /api/predict`
- Create review : `POST /api/review`
- Get reviews : `GET /api/predicts`

**TODO: WRITE SOMETHING ABOUT THE INFERENCE CODE FOR THE SENTIMENT CLASSIFIER**

In order to interact with the database, we will use the ORM [peewee](http://docs.peewee-orm.com/en/latest/). It lets us define the dataset tables using python objects, and takes care of connecting to and querying the database.

This is done in the `src/api/db.py` file:

```python
import peewee as pw
import config

db = pw.PostgresqlDatabase(
    config.POSTGRES_DB,
    user=config.POSTGRES_USER, password=config.POSTGRES_PASSWORD,
    host=config.POSTGRES_HOST, port=config.POSTGRES_PORT
)


class BaseModel(pw.Model):
    class Meta:
        database = db


# Table Description
class Review(BaseModel):

    review = pw.TextField()
    rating = pw.IntegerField()
    suggested_rating = pw.IntegerField()
    sentiment_score = pw.FloatField()
    brand = pw.TextField()
    user_agent = pw.TextField()
    ip_address = pw.TextField()

    def serialize(self):
        data = {
            'id': self.id,
            'review': self.review,
            'rating': int(self.rating),
            'suggested_rating': int(self.suggested_rating),
            'sentiment_score': float(self.sentiment_score),
            'brand': self.brand,
            'user_agent': self.user_agent,
            'ip_address': self.ip_address
        }

        return data


# Connection and table creation
db.connect()
db.create_tables([Review])

```

Having done all this using peewee now makes it super easy to define the api routes to save and get reviews:

```python
import db

@api.route('/review', methods=['POST'])
def post_review():
    '''
    Save review to database.
    '''
    if request.method == 'POST':
        expected_fields = [
            'review',
            'rating',
            'suggested_rating',
            'sentiment_score',
            'brand',
            'user_agent',
            'ip_address'
        ]
        if any(field not in request.form for field in expected_fields):
            return jsonify({'error': 'Missing field in body'}), 400

        query = db.Review.create(**request.form)

        return jsonify(query.serialize())


@api.route('/reviews', methods=['GET'])
def get_reviews():
    '''
    Get all reviews.
    '''
    if request.method == 'GET':
        query = db.Review.select()

        return jsonify([r.serialize() for r in query])

```

Now we can have a closer look at the routes' request bodies and responses.

**Sentiment Classification Route**

Route used to predict the sentiment based on the review's text.

Body:
```python
{
    "review": "I hate this brand..."
}
```

Response:
```
0.123
```

**Create Review**

Route used to save a review to database (with associated ratings and user information).

Body:
```python
{
    "review": "I hate this brand...",
    "rating": 2,
    "suggested_rating": 1,
    "sentiment_score": 0.123,
    "brand": "Apple",
    "user_agent": "Mozilla/...",
    "ip_address": "127.0.0.1"
}
```

Response:
```python
{
  "id": 123,
  "review": "I hate this brand...",
  "rating": 2,
  "suggested_rating": 1,
  "sentiment_score": 0.123,
  "brand": "Apple",
  "user_agent": "Mozilla/...",
  "ip_address": "127.0.0.1"
}
```

**Get Reviews**

Route used to get reviews.

Response:
```python
[
    {
        "id": 1234,
        "first_name": "Joe",
        "last_name": "Bloggs",
        "email": "joe25@example.com",
        "uapp": "ios1_2"
    }
]
```


## Dockerizing the application with Docker compose 🐳

--> Marwan

## Deploying to AWS: Demo time 💻

--> Marwan

## Where to go from here 🏍

[Random ideas thrown in random orders]

- How to manage multiple concurrent sessions
- Deploy on multiple EC2 machines 
- Use CI/CD 
- Use Kubernetes to manage clusters of containers

## Contributions and pull requests  🛠

This would be awesome !
