# End 2 End Machine Learning : From Data Collection to Deployment

## Introduction

In this post, we'll go through the necessary steps to build and deploy a machine learning application. This starts from data collection to deployment and the journey, as you'll see it, is exciting and fun. 

Before to start, let's look first a the app we'll be building:

<!-- insert GIF or VIDEO here -->

As you see it, this web app allows a user to evaluate random brands by writing reviews. While writing, the user will see the sentiment score of his input in real-time along with a proposed rating from 1 to 5.

The user can then fix the rating and submit.

You can think of this as a crowd sourcing app of brand reviews with a sentiment analysis model that suggests ratings.

## Scraping the data from Trustpilot

In order to train a sentiment classifier, we need data. We can sure download open source datasets for sentiment analysis tasks such as Amazon polarity or IMDB movie reviews but for the purpose of this tutorial, **we'll build our own dataset**. 

To collect labeled data in order to train a sentiment classifier, we'll scrape customer reviews from Trustpilot. Trustpilot.com is a consumer review website founded in Denmark in 2007 and hosts reviews of businesses worldwide. Nearly 1 million new reviews are posted each month.

<img src="./assets/truspilot.png">

In this post, Wwe'll focus on english reviews only. 

Trustpilot is an interesting source because each customer review is associated with a number of stars. By leveraging this data, we can infer a sentiment label for each review.

<p style="text-align: center">
  <img src="./assets/review_label.png" width="50%">
</p>

We mapped each review to a class based on the number of stars and we used this information for training the sentiment classifier.

- 1 and 2 stars: bad reviews
- 3 stars: average reviews
- 4 and 5 stars: good reviews


In order to scrape customer reviews from trustpilot, we have to first understand the structure of the website. 

Trustpilot is organized by categories of businesses.

<img src="./assets/1-categories.png">

Each category is divided into sub-categories.

<img src="./assets/2-subcategories.png">

Each sub-category is divided into companies.

<img src="./assets/3-companies.png">

And then each companies has its own set of reviews. 

<img src="./assets/4-reviews.png">

Pretty simple right?


## Training a sentiment classifer usig PyTorch

## Building an interactive web interface with Dash, Flask and Post



## Dockerizing the application with Docker compose

## Deploying to AWS: Demo time

## Where to go from here?
