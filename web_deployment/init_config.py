import os
import configparser
import shutil

# If using aws for experiment tracking and model registry fill in with the public DNS of the EC2 instance, else leave it ""
TRACKING_SERVER_HOST = "ec2-18-212-173-4.compute-1.amazonaws.com" # fill in with the public DNS of the EC2 instance or leave it ""

# If using aws for experiment tracking and model registry fill with your aws profile, else leave it ""
AWS_PROFILE = ""

os.environ["AWS_PROFILE"] = AWS_PROFILE

config = configparser.ConfigParser()
config['DEFAULT'] = {"TRACKING_SERVER_HOST": TRACKING_SERVER_HOST, "AWS_PROFILE": AWS_PROFILE}

with open('./config.config', 'w') as configfile:
    config.write(configfile)

print("Config initialized")