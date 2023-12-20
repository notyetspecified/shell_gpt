FROM python:3-slim

ARG UID=1000
ARG GID=1000

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_ROOT_USER_ACTION ignore
WORKDIR /app
COPY . /app

RUN pip install --no-cache --upgrade pip \
 && pip install --no-cache /app \
 && addgroup --gid $GID --system app \
 && adduser --uid $UID --home /home/app --system --group app \
 && mkdir -p /tmp/shell_gpt \
 && chown -R app:app /tmp/shell_gpt

USER app

ENTRYPOINT ["sgpt"]
