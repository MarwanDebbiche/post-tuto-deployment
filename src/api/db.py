import peewee as pw

db = pw.PostgresqlDatabase(
    'postgres', user='postgres', password='password',
    host='db', port=5432
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
