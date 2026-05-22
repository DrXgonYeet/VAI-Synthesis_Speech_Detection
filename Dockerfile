# Use Python 3.10 to support newer NumPy versions
FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Keep your main:app setting
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]