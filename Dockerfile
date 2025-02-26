FROM python:3.10-slim-buster

# Install necessary libs
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc wget

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install && \
  pip install TA-Lib==0.4.31

# Get time and timezone from host
ENV TZ=Europe/Moscow
RUN apt-get install -yy tzdata
RUN cp /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /sigbot

CMD ["python3", "main.py"]
