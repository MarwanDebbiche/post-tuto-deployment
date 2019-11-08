import peewee as pw
import config

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

    def serialize(self):
        data = {
            'id': self.id,
            'review': self.review,
            'rating': int(self.rating),
            'suggested_rating': int(self.suggested_rating),
            'sentiment_score': float(self.sentiment_score),
            'brand': self.brand,
            'user_agent': self.user_agent,
            'ip_address': self.ip_address
        }

        return data


# Connection and table creation
db.connect()
db.create_tables([Review])
