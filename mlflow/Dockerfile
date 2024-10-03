FROM python:3.11

COPY requirements.txt  /workdir/

WORKDIR /workdir
RUN pip install -r requirements.txt

COPY . .

EXPOSE 600

CMD [ "mlflow", "server","--host","0.0.0.0","--port","600"]