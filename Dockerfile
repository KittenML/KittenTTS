# Use an official Python runtime as a parent image
# Using a slim version keeps the image size smaller
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install the 'wheel' package, which is needed to build the wheel file
RUN pip install setuptools wheel

# Copy the necessary project files into the container
# This includes the setup.py and the script it references
COPY setup.py README.md .

# Build the wheel file and then install it with pip
# This command first creates the wheel in the 'dist' folder and then installs it
# Installing from the wheel is a clean and efficient way to handle dependencies
RUN python setup.py bdist_wheel && \
    pip install dist/kittentts-0.1.0-py3-none-any.whl

COPY . .

RUN python app.py --init

# Define the default command to run when the container starts
# The entrypoint will be the Python script itself
ENTRYPOINT ["python", "app.py"]

# The command is left empty so the user can pass arguments directly
# to the script when running the container (e.g., "docker run ... 'Hello, world!'")
CMD []
