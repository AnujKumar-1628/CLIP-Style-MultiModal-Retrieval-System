# Docker

This folder contains the container image and Compose configuration for the
retrieval API and project pipelines.

## Files

- `Dockerfile`: CPU-oriented Python image shared by the API and pipelines.
- `docker-compose.yml`: local API service, persistent mounts, and health check.
- `entrypoint.sh`: named commands for the API, training, indexing, and evaluation.

The build context is the repository root so `.dockerignore` also lives there.

## API

Run all commands from the repository root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The API is available at `http://localhost:8001`, with docs at
`http://localhost:8001/docs`.

Override the host port when needed:

```bash
API_PORT=9000 docker compose -f docker/docker-compose.yml up
```

## Pipelines

The Compose service reuses the same image for one-off jobs:

```bash
docker compose -f docker/docker-compose.yml run --rm api train
docker compose -f docker/docker-compose.yml run --rm api index
docker compose -f docker/docker-compose.yml run --rm api evaluate
```

Arguments after the named command are passed to the corresponding script:

```bash
docker compose -f docker/docker-compose.yml run --rm api index \
  configs/retrieval.yaml configs/model.yaml configs/data.yaml test
```

`data`, `artifacts`, and `experiments` are bind-mounted from the repository.
The Hugging Face model cache is stored in a named Docker volume.

The extra read-only `/content/local_data/data` mount supports the current
absolute Flickr30k paths in `configs/data.yaml`.
