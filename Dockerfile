FROM python:alpine

WORKDIR /app

RUN pip install pandas==1.4.3
RUN pip install numpy==1.23.2
RUN pip install ta-lib==0.4.19
RUN pip install python-binance==1.0.16
RUN pip install python-dotenv==0.20.0
RUN pip install mplfinance==0.12.9b1
RUN pip install matplotlib==3.5.3
RUN pip install python-telegram-bot==13.13
RUN pip install proplot==0.9.5

RUN sudo apt-get install msttcorefonts

COPY . .

CMD ["python", "main.py"]