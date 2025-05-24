# Using the official Python image as the base image
FROM python:3.9

# Setting the working directory inside the container
WORKDIR /app

# Copying the FastAPI application files into the container
COPY main.py /app/main.py
COPY my_model /app/my_model

# Installing any necessary dependencies
RUN pip install fastapi uvicorn pillow numpy tensorflow python-multipart

# Exposing the port the application will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# To build a docker image run the following command on your terminal
# docker build -t my_fastapi_app

# To run the docker image you just built run the following command
# docker run -p 8000:8000 my_fastapi_app

# After this you can find the api on http://127.0.0.1:8000/docs

# When you're done docker stop <container-id> you can have the container id by docker ps which shows you all the running containers
