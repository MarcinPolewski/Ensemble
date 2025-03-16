import requests


TOKEN = "Yz73gVpLKKZyyCqywhflwg66vGQkel"                         # Your token here
URL = "http://149.156.182.9:6060/task-5/submit"
agent_file = "/net/tscratch/people/tutorial030/Ensemble/task_5/agent.py"

# These are just some random .pt files as an example
weights_file = '/net/tscratch/people/tutorial030/Ensemble/task_5/target_model.pt'


def submitting_example():
    with open(agent_file, "rb") as agent, open(weights_file, "rb") as weight:
        files = [
            ("agent_file", ("agent.py", agent, "application/octet-stream")),
            ("files", ("target_model.pt", weight, "application/octet-stream")),
            # ... You can add up to 5 files with weights here
        ]

        result = requests.post(
            URL,
            headers={"token": TOKEN},
            files=files
        )

        print(result.status_code, result.text)


if __name__ == '__main__':
    submitting_example()
