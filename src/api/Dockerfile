FROM python:3.6

ADD requirements.txt /app/
WORKDIR /app

RUN pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

ADD . /app

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["app.py"]
