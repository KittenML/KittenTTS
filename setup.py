from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kittentts",
    version="0.1.0",
    author="KittenML",
    author_email="",
    description="Ultra-lightweight text-to-speech model with just 15 million parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kittenml/kittentts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "num2words",
        "spacy>=3.8.0",
        "espeakng_loader",
        "misaki>=0.7.4",
        "phonemizer",
        "onnxruntime",
        "soundfile",
        "numpy>=1.26.0",
        "huggingface_hub",
    ],
    keywords="text-to-speech, tts, speech-synthesis, neural-networks, onnx",
    project_urls={
        "Bug Reports": "https://github.com/kittenml/kittentts/issues",
        "Source": "https://github.com/kittenml/kittentts",
    },
)
