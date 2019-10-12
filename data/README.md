In this section we develop a scraper that fetches customer reviews from https://fr.trustpilot.com/

We proceed as the following:

- We start by fetching the company profiles urls on trustpilot using Selenium. 

This results in the file: scraping/notebooks/exports/consolidate_company_urls.csv. See notebook scraping/notebooks/scrape_website_urls.ipynb for details

- We go over each company and scrape its reviews.

To run the script 

```bash
cd scraping/
scrapy crawl trustpilot
```

This results in the comments_trustpilot.json file under scraping/