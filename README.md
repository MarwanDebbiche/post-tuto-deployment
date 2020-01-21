# End 2 End Machine Learning : From Data Collection to Deployment 🚀 

In this job, I collaborated with <a href="https://github.com/ahmedbesbes">Ahmed BESBES</a>

Medium post <a href="https://medium.com/@ahmedbesbes/end-to-end-machine-learning-from-data-collection-to-deployment-ce74f51ca203">here</a>.

You may also read about it <a href="https://ahmedbesbes.com/end-to-end-ml.html">here</a> and <a href="https://marwandebbiche.com/posts/e2e-ml/">here</a>.

In this post, we'll go through the necessary steps to build and deploy a machine learning application. This starts from data collection to deployment; and the journey, you'll see, is exciting and fun. 😀

Before we begin, let's have a look at [the app](https://www.reviews.ai2prod.com/) we'll build:

<p align="center">
    <img src="./blog-post/assets/app.gif"  style="margin:15px">
</p>

As you see, this web app allows a user to evaluate random brands by writing reviews. While writing, the user will see the sentiment score of his input updating in real-time, alongside a proposed 1 to 5 rating.

The user can then change the rating in case the suggested one does not reflect his views, and submit.

You can think of this as a crowd sourcing app of brand reviews, with a sentiment analysis model that suggests ratings that the user can tweak and adapt afterwards.

To build this application, we'll follow these steps:

- Collecting and scraping customer reviews data using `Selenium` and `Scrapy`
- Training a deep learning sentiment classifier on this data using `PyTorch`
- Building an interactive web app using `Dash`
- Setting a `REST API` and a `Postgres` database
- Dockerizing the app using `Docker Compose`
- Deploying to `AWS`

<hr>

## Project architecture 

### Run the app locally


To run this project locally using `Docker Compose` `run`: 

```
docker-compose build
docker-compose up
```
You can then access the dash app at [http://localhost:8050](http://localhost:8050)

### Development

If you want to contribute to this project and run each service independently:

#### Launch API

In order to launch the API, you will first need to run a local `postgres` db using `Docker`:

```
docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres -p 5432:5432 -d postgres
```

Then you'll have to type the following commands:

```shell
cd src/api/
python app.py
```

#### Launch Dash app

In order to run the `dash` server to visualize the output:

```shell
cd src/dash/
python app.py
```


### How to contribute 😁

Feel free to contribute! Report any bugs in the [issue section](https://github.com/MarwanDebbiche/post-tuto-deployment/issues).

Here are the few things we noticed, and wanted to add.

- [ ] Add server-side pagination for Admin Page and `GET /api/reviews` route.
- [ ] Protect admin page with authentication.
- [ ] Either use [Kubernetes](https://kubernetes.io) or [Amazon ECS](https://aws.amazon.com/ecs) to deploy the app on a cluster of containers, instead of on one single EC2 instance.
- [ ] Use continuous deployment with [Travis CI](https://travis-ci.org)
- [ ] Use a managed service such as [RDD](https://aws.amazon.com/rds/) for the database


### Licence

MIT
