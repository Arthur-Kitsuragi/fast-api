## Usage

- git clone https://github.com/Arthur-Kitsuragi/fast-api.git

- cd fast-api

- docker build -t fastapi-pdf-classifier .

- docker run -d --gpus all -p 5000:5000 fastapi-pdf-classifier

- open your browser and go to http://localhost:5000/docs

- locust -f locust.py --host=http://your_ip  (locust)

- pytest -v tests/ (another tests)