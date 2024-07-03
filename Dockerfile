# pull python base image
FROM python:3.10

# specify working directory
#WORKDIR 

ADD /requirements.txt .
ADD /*.* .
ADD /*.txt .
ADD /pages/* ./pages/

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

#RUN rm *.whl

# copy application files
#ADD /heart_model/* ./app/

# expose port for application
EXPOSE 8001

# start fastapi application
ENTRYPOINT ["streamlit", "run", "welcome.py", "--server.port=8001", "--server.address=0.0.0.0"]
