In this section we develop a scraper that fetches customer reviews from https://fr.trustpilot.com/

We proceed as following:

1. We start by fetching the company profiles urls on trustpilot using Selenium. 

This results in the file: src/scraping/selenium/exports/consolidate_company_urls.csv. 

See notebook src/scraping/selenium/scrape_website_urls.ipynb for details

2. We go over each company and scrape its reviews using Scrapy.

To run the script 

```shell
cd src/scraping/scrapy/
scrapy crawl trustpilot
```

This results in the comments_trustpilot.csv file under src/scraping/scrapy/
