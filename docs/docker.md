# Running with Docker

Use a Linux host with Docker, an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The host driver must meet the [vLLM GPU requirements](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/).

From the repository root, build the image:

```sh
docker build -t kittenml .
```

Save the [README's vLLM example](../README.md#running-with-vllm) as
`vllm_example.py`, then run it:

```sh
docker run --rm --gpus all --ipc=host \
  -v kittenml-cache:/cache -v "$PWD:/workspace" \
  kittenml python vllm_example.py
```

The command saves `output.wav` in the current directory. The named volume keeps
model downloads and compilation caches between runs. Model files are downloaded
on first use and are not included in the image.

The image also includes the existing PyTorch backend. Run your own Python script
by replacing `vllm_example.py` in the command above.
