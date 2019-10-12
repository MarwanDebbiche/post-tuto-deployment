import re
import scrapy

class Pages(scrapy.Spider):
    name = "trustpilot"

    start_urls = [
        "https://fr.trustpilot.com/categories"
    ]

    def parse(self, response):
        for category in response.css('div.category-object'):
            uris = category.css('div.sub-category-list a.sub-category-item::attr(href)').extract()
            names = category.css('div.sub-category-list a.sub-category-item::text').extract()
            for name, uri in zip(names, uris):
                request = response.follow(uri, callback=self.parse_websites_page)
                request.meta['category'] = name
                yield request 

    def parse_websites_page(self, response):
        for a in response.css('div.rankings h2 a'):
            uri = a.css('::attr(href)').extract_first().strip()
            request = response.follow(uri, callback=self.parse_comments)
            request.meta['url_website'] = uri
            request.meta['url_category'] = response.meta['url_category']
            yield request

        next_page = response.css('a[data-page-number=next-page] ::attr(href)').extract_first()
        if next_page is not None:
            request = response.follow(next_page, callback=self.parse_websites_page)
            request.meta['url_category'] = response.meta['url_category']
            yield request


    def parse_comments(self, response):         
        for section in response.css('section.review-card__content-section'):
            comment = section.css("div.review-info__body p.review-info__body__text::text").extract_first().strip()
            rating = section.css("div.review-info__header__verified div::attr(class)").extract_first()
            rating = int(re.search(r'\d', rating).group(0))
            if rating > 3:
                label = 1
            else:
                label = 0

            try:
                url_website = response.meta['url_website'].split('/')[-1]
            except:
                url_website = None
        
            try:
                url_category = response.meta['url_category'].split('/')[-1]
            except:
                url_category = None
            
            yield {
                "text": comment,
                "rating": rating,
                "label": label,
                "url_website": url_website,
                "url_category": url_category

            }

        next_page = response.css('a[data-page-number=next-page] ::attr(href)').extract_first()
        if next_page is not None:
            request = response.follow(next_page, callback=self.parse_comments)
            request.meta['url_category'] = response.meta['url_category']
            request.meta['url_website'] = response.meta['url_website']
            yield request
            
