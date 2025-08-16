# DIAGNOSTIC DOCKERFILE: To discover pre-installed packages in the DLC.

# Use the correct, private ECR path for the AWS DLC image
FROM 763124950121.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310

# The only command is to run pip freeze and save the output.
RUN pip freeze > /tmp/dlc_requirements.txt

CMD ["echo", "Discovery complete. File is at /tmp/dlc_requirements.txt"]
