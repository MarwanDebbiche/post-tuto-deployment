# End 2 End Machine Learning : From Data Collection to Deployment

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

You can then launch the API by running the following commands:

```
cd src/dash
python app.py
```
