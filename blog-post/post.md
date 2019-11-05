# End 2 End Machine Learning : From Data Collection to Deployment

## Introduction

In this post, we'll go through the necessary steps to build and deploy a machine learning application. This starts from data collection to deployment and the journey, as you'll see it, is exciting and fun. 

Before to start, let's first look at the app we'll be building:

<!-- insert GIF or VIDEO here -->

As you see it, this web app allows a user to evaluate random brands by writing reviews. While writing, the user will see the sentiment score of his input in real-time along with a proposed rating from 1 to 5.

The user can then fix the rating and submit.

You can think of this as a crowd sourcing app of brand reviews with a sentiment analysis model that suggests ratings.

## Scraping the data from Trustpilot with Selenium and Scrapy

In order to train a sentiment classifier, we need data. We can sure download open source datasets for sentiment analysis tasks such as Amazon polarity or IMDB movie reviews but for the purpose of this tutorial, **we'll build our own dataset**. 

To collect labeled data in order to train a sentiment classifier, we'll scrape customer reviews from Trustpilot. Trustpilot.com is a consumer review website founded in Denmark in 2007 and hosts reviews of businesses worldwide. Nearly 1 million new reviews are posted each month.

<p align="center">
  <img src="./assets/truspilot.png" width="90%">
</p>

In this post, Wwe'll focus on english reviews only. 

Trustpilot is an interesting source because each customer review is associated with a number of stars. By leveraging this data, we can infer a sentiment label for each review.

<p align="center">
  <img src="./assets/review_label.png" width="70%">
</p>

We mapped each review to a class based on the number of stars and we used this information for training the sentiment classifier.

- 1 and 2 stars: bad reviews
- 3 stars: average reviews
- 4 and 5 stars: good reviews


In order to scrape customer reviews from trustpilot, we have to first understand the structure of the website. 

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

And then each companies has its own set of reviews. 

<p align="center">
  <img src="./assets/4-reviews.png" width="80%">
</p>

As you see, this is a top down tree structure. In order to scrape it efficiently we'll use **Scrapy** framework, but before going that far we need a little bit of Selenium to fetch the company urls first, then feed those to Scrapy.

We unfortunately need to use Selenium because the content of the website that renders those urls is dynamic (but the rest is not) and cannot be accessed from the page source like Scrapy does. Selenium simulates a browser that clicks on each category, narrows down to each sub-category and finally goes through all the companies one by one and fetches their urls. When it's done, the script saves these urls to a csv file and the Scrapy part can be launched.

Let's see how to launch Selenium to fetch the company urls:

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


We start by fetching the sub-categories URIs nested inside each category:

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

This function allows to fetch company urls referenced in a given subcategory:

```python
def extract_company_urls_form_page():
    a_list = driver.find_elements_by_xpath('//a[@class="category-business-card card"]')
    urls = [a.get_attribute('href') for a in a_list]
    dedup_urls = list(set(urls))
    return dedup_urls
```

and this one indicates if a pagination exists on the page:

```python
def go_next_page():
    try:
        button = driver.find_element_by_xpath('//a[@class="button button--primary next-page"]')
        return True, button
    except NoSuchElementException:
        return False, None
```

Now we initialize Selenium with a headless Chromedriver:

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

and launch the scraping. This approximatively takes 50 minutes.

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



## Training a sentiment classifer usig PyTorch


## Building an interactive web interface with Dash, Flask and PostgeSQL

--> provide a diagram for the architecture to have a global picture first

## Dockerizing the application with Docker compose

--> Marwan

## Deploying to AWS: Demo time

--> Marwan

## Where to go from here?

[Random ideas thrown in random orders]

- How to manage multiple concurrent sessions
- Deploy on multiple EC2 machines 
- Use CI/CD 
- Use Kubernetes to manage clusters of containers 

## Contributions and pull requests

This would be awesolme 😁