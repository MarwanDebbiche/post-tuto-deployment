# End to End Machine Learning : From Data Collection to Deployment

## Introduction

In this post, we'll go through the necessary steps to build and deploy a machine learning application. This starts from data collection to deployment and the journey, as you'll see it, is exciting and fun. 😀

Before to start, let's first look at the app we'll be building:

<p align="center">
    <img src="./assets/app.gif">
</p>

As you see it, this web app allows a user to evaluate random brands by writing reviews. While writing, the user will see the sentiment score of his input updating in real-time along with a proposed rating from 1 to 5.

The user can then fix the rating and submit.

You can think of this as a crowd sourcing app of brand reviews with a sentiment analysis model that suggests ratings which the user can tweak and adapt.

To build this application we'll follow the following steps:

- Data collection
- Model training
- App development
- Containerization
- App deployment

All the code is available in github and organized in independant directories.

Let's get started! 👨🏻‍💻

## Scraping the data from Trustpilot with Selenium and Scrapy ⛏

In order to train a sentiment classifier, we need data. We can sure download open source datasets for sentiment analysis tasks such as Amazon polarity or IMDB movie reviews but for the purpose of this tutorial, **we'll build our own dataset**. We'll scrape customer reviews from Trustpilot. 

Trustpilot.com is a consumer review website founded in Denmark in 2007. It hosts reviews of businesses worldwide and nearly 1 million new reviews are posted each month.


<p align="center">
  <img src="./assets/truspilot.png" width="90%">
</p>


Trustpilot is an interesting source because each customer review is associated with a number of stars.

<p align="center">
  <img src="./assets/review_label.png" width="70%">
</p>

By leveraging this data, we are able to map each review to a sentiment class based on its number of stars so that reviews with:

- 1 and 2 stars are **bad reviews**
- 3 stars are **average reviews**
- 4 and 5 stars are **good reviews**


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


As you see, this is a top down tree structure. In order to scrape it efficiently we'll use **Scrapy** framework, but before going that far we need a little bit of Selenium to fetch the company urls first (see previous screenshot), then feed those to Scrapy.

We unfortunately need to use Selenium because the content of the website that renders those urls is dynamic (but the rest is not) and cannot be accessed from the page source like Scrapy does. Selenium simulates a browser that clicks on each category, narrows down to each sub-category and finally goes through all the companies one by one and fetches their urls. When it's done, the script saves these urls to a csv file and the Scrapy part can be launched.

### Collect company urls with Selenium


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

If you open up your browser and inspect the source code, you'll find out 22 category blocks (on the right) located in ```div``` objects that have a ```class``` attribute equal to  ```category-object```

<p align="center">
    <img src="./assets/5-category_block.png" width="80%">
</p>

Each category has its own set of sub-categories. Those are located in ```div``` objects that have ```class``` attributes equal to ```child-category```.
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

Once the scraping is over, we save the urls to a csv file:

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

### Scraping customer reviews with Scrapy 

Ok, now we're ready to scrape the data we need with Scrapy.

First you need to make sure Scrapy is installed. Otherwise, you can install using
- conda: ```conda install -c conda-forge scrapy``` 

or 

- pip: ```pip install scrapy```

Then, you'll need to start a scrapy project:

```bash
cd src/scraping/scrapy
scrapy startproject trustpilot
```

This last command creates a structure of a Scrapy project. Here's what it looks like:

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

To build our scraper we'll create a spider inside the ```spiders``` folder.

What the scraper basically does is the following:

- It starts from a company url
- It goes through each customer review and extracts a dictionary of data cotaining the following items

    - comment: the text review
    - rating: the number of stars (1 to 5)
    - url_website: the company website on trustpilot
    - company_name: the company being reviewed
    - company_website: the website of the company being reviewed
    - company_logo: the logo of the company being reviewed   
- It moves to the next page if any

Here's the full script:

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

Before launching the scraper you have to change the settings.py:

Here are the changing we made:

```python
# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 32

#Export to csv
FEED_FORMAT = "csv"
FEED_URI = "comments_trustpilot_en.csv"
```
This indicates to the scraper to ignore robots.txt, to use 32 concurrent requests and to export the data into a csv format under the filename: ```comments_trustpilot_en.csv```

Now time to launch the scraper: 

```bash 
cd src/scraping/scrapy
scrapy crawl trustpilot
```

We'll let it run for a little bit of time.

Note that we can interrupt it at any moment since it saves the data on the fly.

<u>**Disclaimer :**</u> 

This script is meant for educational purposes only: scrape responsively.

## Training a sentiment classifer usig PyTorch 🤖

Now the data is collected and we're ready to train a sentiment classifier.



## Building an interactive web app 📲 with Dash, Flask and PostgeSQL 

--> provide a diagram for the architecture to have a global picture first

## Dockerizing the application with Docker compose 🐳

--> Marwan

## Deploying to AWS: Demo time 🚀

--> Marwan

## Where to go from here ❓

[Random ideas thrown in random orders]

- How to manage multiple concurrent sessions
- Deploy on multiple EC2 machines 
- Use CI/CD 
- Use Kubernetes to manage clusters of containers

## Contributions and pull requests 🔧

This would be awesolme 😁
