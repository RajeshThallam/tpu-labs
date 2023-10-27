
Configure the project and zone
```bash
export PROJECT_ID=rthallam-demo-project
export ZONE=us-central1-b

gcloud alpha compute tpus tpu-vm service-identity create --zone=${ZONE}

gcloud auth login
gcloud config set project ${PROJECT_ID}
gcloud auth application-default set-quota-project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
```



```bash
export PROJECT_ID=tpu-prod-env-small
export ZONE=us-east1-c
export ACCELERATOR_TYPE=v5litepod-1
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
export SERVICE_ACCOUNT=463402977885-compute@developer.gserviceaccount.com
export TPU_NAME=rthallam-v5e-1
export QUEUED_RESOURCE_ID=rthallam-tpu-20231025
```

```bash
gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} \
  --node-id ${TPU_NAME} \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --accelerator-type ${ACCELERATOR_TYPE} \
  --runtime-version ${RUNTIME_VERSION} \
  --service-account ${SERVICE_ACCOUNT} \
  --reserved
```

```bash
gcloud alpha compute tpus queued-resources describe rthallam-tpu-20231024 --format='value(state.state)'
```

```bash
export PROJECT_ID=rthallam-demo-project
export ZONE=europe-west4-a
export ACCELERATOR_TYPE=v3-8
export TPU_NAME= rt-tpu-v3-8
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --accelerator-type ${ACCELERATOR_TYPE} \
  --version tpu-ubuntu2204-base
```