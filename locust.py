from locust import HttpUser, task, between

class PDFUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def upload_pdf(self):
        with open("tests/resources/test.pdf", "rb") as f:
            files = {"files": ("test.pdf", f, "application/pdf")}

            response = self.client.post("/uploadfiles", files=files)

            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
