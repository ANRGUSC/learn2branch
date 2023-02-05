# Coder Tempalte for Learn2Branch

## Build the devcontainer image
To build the devcontainer image, run the following commands:
```bash
# cd to the devcontainer directory
cd learn2branch/.devcontainer
# Build the devcontainer image
docker build -t jaredraycoleman/learn2branch:latest .. -f Dockerfile
## Push the devcontainer image
docker push jaredraycoleman/learn2branch:latest
```

You can replace the docker repository with your own.
If you do, make sure to replace in in [./main.tf](./main.tf) as well.

## Coder Template
To push the pre-configured Coder template, run the following commands:
```bash
# Create the template
coder templates create learn2branch # run this only the first time
# Update the template
coder templates push learn2branch # run this every time you update the template
```
