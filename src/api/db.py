import peewee as pw
import config
from datetime import datetime
from playhouse.shortcuts import model_to_dict


db = pw.PostgresqlDatabase(
    config.POSTGRES_DB,
    user=config.POSTGRES_USER, password=config.POSTGRES_PASSWORD,
    host=config.POSTGRES_HOST, port=config.POSTGRES_PORT
)


class BaseModel(pw.Model):
    class Meta:
        database = db


# Table Description
class Review(BaseModel):

    review = pw.TextField()
    rating = pw.IntegerField()
    suggested_rating = pw.IntegerField()
    sentiment_score = pw.FloatField()
    brand = pw.TextField()
    user_agent = pw.TextField()
    ip_address = pw.TextField()
    created_date = pw.DateTimeField(default=datetime.now)

    def serialize(self):
        review_dict = model_to_dict(self)
        review_dict["created_date"] = (
            review_dict["created_date"].strftime('%Y-%m-%d %H:%M:%S')
        )

        return review_dict


# Connection and table creation
db.connect()
db.create_tables([Review])
