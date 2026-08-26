# KittenTTS Docker Image

## Building the Image

To build the Docker image, run the following command from the root of the project:

```bash
docker build -t kittentts -f .docker/Dockerfile .
```

## Running the Container

To generate audio, you can run the Docker container with the following command.
This example will create an `output` directory on your host if it doesn't exist and save the generated audio file there.

```bash
docker run --rm kittentts \
    -v "$(pwd)/output:/app/output" \ 
    --text "Hello world, this audio was generated from inside a Docker container." \
    --output "/app/output/hello_docker.wav"
```
