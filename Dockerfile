FROM python:3.12-slim-bookworm

# Install necessary libs
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  wget \
  tzdata \
  libgomp1 && \
  rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install TA-Lib
# RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
#   tar -xvzf ta-lib-0.6.4-src.tar.gz && \
#   cd ta-lib/ && \
#   ./configure --prefix=/usr && \
#   make && \
#   make install && \
#   pip install TA-Lib==0.6.4
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb && \
  dpkg -i ta-lib_0.6.4_amd64.deb && \
  pip install TA-Lib==0.6.4

# Get time and timezone from host
ENV TZ=Europe/Moscow
RUN cp /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /sigbot

CMD ["python3", "main.py"]
