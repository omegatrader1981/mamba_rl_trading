# DIAGNOSTIC DOCKERFILE: To discover pre-installed packages in the DLC.

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310

# The only command is to run pip freeze and save the output.
RUN pip freeze > /tmp/dlc_requirements.txt

# Add a final command to keep the container alive for a moment if needed.
# This is useful for local debugging but not strictly necessary for CodeBuild.
CMD ["echo", "Discovery complete. File is at /tmp/dlc_requirements.txt"]
