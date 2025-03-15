import requests
import yaml
import os
import time
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
APPLICATION_ID = os.getenv("APPLICATION_ID")
URL = f"https://discord.com/api/v9/applications/{APPLICATION_ID}/commands"


with open("discord_commands.yaml", "r") as file:
    yaml_content = file.read()

commands = yaml.safe_load(yaml_content)
headers = {
    "Authorization": f"Bot {TOKEN}",
    "Content-Type": "application/json"
}

# Send the POST request for each command
for command in commands:
    while True:
        response = requests.post(URL, json=command, headers=headers)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", 1)
            print(f"Rate-limited. Retrying after {retry_after} seconds.")
            time.sleep(float(retry_after))
        else:
            command_name = command["name"]
            print(f"Command '{command_name}' created: {response.status_code}")
            break