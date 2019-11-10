# End 2 End Machine Learning : From Data Collection to Deployment 🚀 

Repository for the blog post shared between [Ahmed Besbes](http://ahmedbesbes.com) and [Marwan Debbiche](http://marwandebbiche.com)

To run this project locally, run : 

```
docker-compose build
docker-compose up
```
You can then access the dash app at [http://localhost:8050](http://localhost:8050)

## Development

### Launch API

In order to launch the API, you will first need to run a local postgres db using Docker:

```
docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres -p 5432:5432 -d postgres
```

Then you'll have to type the following commands:

- to run the api: for database interaction and model serving

```shell
cd src/api/
python app.py
```

- to run the dash server: for visualizing the web app

```shell
cd src/dash/
python app.py
```
