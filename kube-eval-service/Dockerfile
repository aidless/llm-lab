ARG PYTHON_BASE_IMAGE=python:3.12-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG INSTALL_DEPENDENCIES=1
RUN if [ "$INSTALL_DEPENDENCIES" = "1" ]; then \
      pip install --no-cache-dir \
        "fastapi==0.162.0" \
        "uvicorn==0.54.0" \
        "httpx==0.28.1"; \
    elif [ -d /root/.local/lib ]; then \
      site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
      cp -a /root/.local/lib/python*/site-packages/. "$site_packages/"; \
    fi

COPY kube_eval_service ./kube_eval_service

ENV PYTHONPATH=/app

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin eval \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 8080

CMD ["python", "-m", "kube_eval_service.app"]
