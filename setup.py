from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kittentts",
    version="0.8.2",
    author="KittenML",
    author_email="",
    description="Ultra-lightweight text-to-speech models powered by native inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KittenML/KittenTTS",
    packages=find_packages(),
    include_package_data=True,
    package_data={"kittentts": ["*.json", "*.txt", "model_defs/*.json"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "espeakng_loader",
        "phonemizer",
        "kitten-inference==0.1.1",
        "soundfile",
        "numpy",
        "huggingface_hub",
    ],
    keywords="text-to-speech, tts, speech-synthesis, neural-networks, native-inference, cpp",
    project_urls={
        "Bug Reports": "https://github.com/KittenML/KittenTTS/issues",
        "Source": "https://github.com/KittenML/KittenTTS",
    },
)
