FROM ubuntu

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY . .
SHELL ["/bin/bash", "-c"]
EXPOSE 7860
CMD python3 chat.py --test_gui