# End 2 End Machine Learning : From Data Collection to Deployment

## Introduction

In this post, we'll go through the different steps that allow to build and deploy a machine learning application. This starts from data collection to deployment and the journey, as you'll see it, is exciting and fun. 

Before to start, let's look at the final demo and see what the application looks like:

<!-- insert GIF or VIDEO here -->

As you see it, this web app allows to evaluate random brands by writing reviews. While writing, the user will see the sentiment score of his input in real time along with a proposed rating from 1 to 5.

The user can then fix the rating and submit.

You can think of this as a crowd sourcing app of brand reviews with a sentiment analysis component based on a machine learning model.

## Scraping the data from Trustpilot

In order to train a sentiment classifier, we need data. We can sure download open source datasets for sentiment analysis tasks such as Amazon polarity or IMDB movie reviews but for the purpose of this tutorial, **we'll build our own dataset**. 

To collect labeled data in order to train a sentiment classifier, we'll scrape customer reviews from Trustpilot. Trustpilot.com is a consumer review website founded in Denmark in 2007 and hosts reviews of businesses worldwide. Nearly 1 million new reviews are posted each month.

<img src="./assets/truspilot.png">

We'll focus on english reviews only. 

Trustpilot is an interesting choice because each customer review is associated with a number of stars. By leveraging this data, we can infer a sentiment label for each review.

<img src="./assets/review_label.png">

Here is how we did it:

- 1 and 2 stars: bad reviews (label = 0)
- 3 stars: average reviews (label = 1)
- 4 and 5 stars: good reviews (label = 2)


In order to scrape customer reviews from trustpilot, we have to first understand the structure of the website. 

Trustpilot is organized by categories of businesses.

<img src="./assets/1-categories.png">


## Training a sentiment classifer usig PyTorch

## Building an interactive web interface with Dash, Flask and Post



## Dockerizing the application with Docker compose

## Deploying to AWS: Demo time

## Where to go from here?
