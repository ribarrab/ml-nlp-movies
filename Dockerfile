FROM continuumio/anaconda3

# Copy requirements.txt to /root
COPY ./requirements.txt /requirements.txt

# Install requirements
RUN pip install -r /requirements.txt

# Copy files from the local repository to the container
COPY ./api_working.py /api_working.py
COPY ./models/model.h5 /model.h5
COPY ./vectorizers/vectorizer.pkl /vectorizer.pkl

# Run the application
CMD ["python", "api_working.py", "-p", "3000"]