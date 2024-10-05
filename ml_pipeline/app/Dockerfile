FROM python:3.11

COPY /predict/ /workdir/predict
COPY /app/ /workdir/app
COPY /utils/ /workdir/utils
COPY configs.py /workdir
WORKDIR /workdir

RUN pip install --no-cache-dir --upgrade -r app/requirements.txt
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
