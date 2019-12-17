# We Use an official Python runtime as a parent image
FROM python:3.6
# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1
# create root directory for our project in the container
RUN mkdir /openCVRestDjangoApp/
# Set the working directory to /music_service
WORKDIR /openCVRestDjangoApp/
# Copy the current directory contents into the container at /music_service
ADD . /openCVRestDjangoApp/
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
EXPOSE 7007
# CMD exec gunicorn opencvrestsite.wsgi:application — bind 0.0.0.0:8000 — workers 3
CMD python manage.py runserver 127.0.0.1:7007