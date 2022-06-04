FROM ubuntu:latest

# install dependencies
RUN apt-get update \
  && apt-get install -y wget nodejs npm
RUN wget https://souffle-lang.github.io/ppa/souffle-key.public -O /usr/share/keyrings/souffle-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/souffle-archive-keyring.gpg] https://souffle-lang.github.io/ppa/ubuntu/ stable main" | tee /etc/apt/sources.list.d/souffle.list
RUN apt update \
  && apt install -y souffle

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

# set up Python environment
WORKDIR /app
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

COPY . /app/leakage-analysis/

# set up pyright
WORKDIR /app/leakage-analysis/pyright
RUN npm install --g lerna
RUN npm install 
WORKDIR /app/leakage-analysis/pyright/packages/pyright
RUN npm run build

# set up main analysis
WORKDIR /app/leakage-analysis
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# EXPOSE 8081
ENTRYPOINT ["python3", "-m", "src.main"]
