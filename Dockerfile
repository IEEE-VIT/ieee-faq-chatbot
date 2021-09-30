FROM python:3.8

COPY ./requirements.txt /app/requirements.txt

RUN pip install fastapi uvicorn

RUN pip install -r /app/requirements.txt

COPY ./ /app

EXPOSE 8000

WORKDIR /app

RUN python3 -m nltk.downloader punkt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]