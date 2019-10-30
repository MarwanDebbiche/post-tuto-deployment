import peewee as pw
import config

db = pw.PostgresqlDatabase(
    config.POSTGRES_DB,
    user=config.POSTGRES_USER, password=config.POSTGRES_PASSWORD,
    host=config.POSTGRES_HOST, port=config.POSTGRES_PORT
)


# Table Description
class Review(pw.Model):

    review = pw.TextField()
    rating = pw.IntegerField()
    suggested_rating = pw.IntegerField()

    def serialize(self):
        data = {
            'id': self.id,
            'review': self.review,
            'rating': int(self.rating),
            'suggested_rating': int(self.suggested_rating)
        }

        return data

    class Meta:
        database = db


# Connection and table creation
db.connect()
db.create_tables([Review])
