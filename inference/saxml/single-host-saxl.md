# SAXML Model Serving

## Create TPU VM

### TPU v5e

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
export PROJECT_ID=diesel-patrol-382622
export ZONE=us-west4-a
export ACCELERATOR_TYPE=v5litepod-1
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
export SERVICE_ACCOUNT=314837540096-compute@developer.gserviceaccount.com
export TPU_NAME=rthallam-v5e-1
export QUEUED_RESOURCE_ID=rthallam-tpu-20231030
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

### TPU v4/v3

```bash
export PROJECT_ID=rthallam-demo-project
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-8
export TPU_NAME=rt-tpu-v4-8
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --accelerator-type ${ACCELERATOR_TYPE} \
  --version tpu-vm-v4-base
  # tpu-ubuntu2204-base
```

- Disable os-login on TPU VM for SSH
```bash
curl -X PATCH -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" -d "{metadata: {'enable-oslogin': 'FALSE'}}" https://tpu.googleapis.com/v2/projects/$PROJECT_ID/locations/$ZONE/nodes/$TPU_NAME?updateMask=metadata
```


## Build your own SAX containers

```bash
export PROJECT_ID=rthallam-demo-project
export REGION="us"
export DOCKER_REPO_NAME="ai-infrastructure"
export ARTIFACT_REGISTRY_PREFIX=${REGION}"-docker.pkg.dev/"${PROJECT_ID}"/"${DOCKER_REPO_NAME}

gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create ${DOCKER_REPO_NAME} \
 --repository-format=docker \
 --location=${REGION} \
 --project=${PROJECT_ID} \
 --description="Docker repository for AI infrastructure containers"

gcloud builds submit \
  --config build_sax_containers.yaml \
  --substitutions _ARTIFACT_REGISTRY_PREFIX=$ARTIFACT_REGISTRY_PREFIX \
  --timeout "24h" \
  --machine-type=e2-highcpu-32 \
  --quiet
```

## Start SAX Cluster

### Configure variables
```bash
sudo usermod -a -G docker ${USER}
newgrp docker

gcloud auth configure-docker us-docker.pkg.dev

SAX_ADMIN_SERVER_IMAGE_NAME="us-docker.pkg.dev/cloud-tpu-images/inference/sax-admin-server"
SAX_MODEL_SERVER_IMAGE_NAME="us-docker.pkg.dev/cloud-tpu-images/inference/sax-model-server"
SAX_UTIL_IMAGE_NAME="us-docker.pkg.dev/cloud-tpu-images/inference/sax-util"

SAX_VERSION=v1.1.0

export SAX_ADMIN_SERVER_IMAGE_URL=${SAX_ADMIN_SERVER_IMAGE_NAME}:${SAX_VERSION}
export SAX_MODEL_SERVER_IMAGE_URL=${SAX_MODEL_SERVER_IMAGE_NAME}:${SAX_VERSION}
export SAX_UTIL_IMAGE_URL="${SAX_UTIL_IMAGE_NAME}:${SAX_VERSION}"

export SAX_ADMIN_SERVER_DOCKER_NAME="sax-admin-server"
export SAX_MODEL_SERVER_DOCKER_NAME="sax-model-server"
export SAX_CELL="/sax/test"

export SAX_ADMIN_STORAGE_BUCKET=rt-central1

docker pull ${SAX_ADMIN_SERVER_IMAGE_URL}
docker pull ${SAX_MODEL_SERVER_IMAGE_URL}
docker pull ${SAX_UTIL_IMAGE_URL}
```

### Start SAX Admin server
```bash
docker run \
--name ${SAX_ADMIN_SERVER_DOCKER_NAME} \
-it \
-d \
--rm \
--network host \
--env GSBUCKET=${SAX_ADMIN_STORAGE_BUCKET} \
${SAX_ADMIN_SERVER_IMAGE_URL}
```

```bash
docker logs -f ${SAX_ADMIN_SERVER_DOCKER_NAME}
```


### Start SAX Model server
```bash
docker run \
    --privileged  \
    -it \
    -d \
    --rm \
    --network host \
    --name ${SAX_MODEL_SERVER_DOCKER_NAME} \
    --env SAX_ROOT=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
    ${SAX_MODEL_SERVER_IMAGE_URL} \
       --sax_cell=${SAX_CELL} \
       --port=10001 \
       --platform_chip=tpuv4 \
       --platform_topology=1x1
```

### SAX Converter

```dockerfile
FROM python:3.10

RUN git clone https://github.com/google/saxml
RUN pip3 install accelerate torch transformers
RUN pip3 install paxml==1.2.0
RUN pip3 install "jax[tpu]==0.4.18" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# install git lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs
```

```bash
docker run \
    --privileged  \
    -it \
    --rm \
    --name sax_converter \
    -v /mnt/disks/saxml:/saxml_inf \
    --entrypoint /bin/bash \
    sax_converter

sed -i '67s/: 1/: 4/g' /saxml/saxml/tools/convert_llama_ckpt.py

cd /saxml/saxml/tools

CHECKPOINT_PATH=gs://rt-central1/sax-model-repository/gpt-j-6b/checkpoint_00000000
python3 -m convert_gptj_ckpt --base EleutherAI/gpt-j-6b --pax /saxml_inf/pax_6b

CHECKPOINT_PATH=gs://rt-central1/sax-model-repository/llama2_7b/checkpoint_00000000
python3 -m convert_llama_ckpt --base /saxml_inf/llama2/hf/llama-2-7b --pax /saxml_inf/llama2/pax_7b --model-size 7b

gsutil -m cp -r checkpoint_00000000 ${CHECKPOINT_PATH}

touch commit_success.txt
gsutil cp commit_success.txt ${CHECKPOINT_PATH}/
gsutil cp commit_success.txt ${CHECKPOINT_PATH}/metadata/
gsutil cp commit_success.txt ${CHECKPOINT_PATH}/state/
```

### Publish model

- GPT-J
```bash
MODEL_NAME=gptjtokenizedbf16bs32
MODEL_CONFIG_PATH=saxml.server.pax.lm.params.gptj.GPTJ4TokenizedBF16BS32
REPLICA=1
CHECKPOINT_PATH=gs://rt-central1/sax-model-repository/gpt-j-6b/checkpoint_00000000
```

- Llama2-7B
```bash
MODEL_NAME=llama7bfp16tpuv4
MODEL_CONFIG_PATH=saxml.server.pax.lm.params.lm_cloud.LLaMA7BFP16TPUv4
REPLICA=1
CHECKPOINT_PATH=gs://rt-central1/sax-model-repository/llama2_7b/checkpoint_00000000
```

- Publish model
```bash
docker run \
 ${SAX_UTIL_IMAGE_URL} \
   --sax_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
   publish \
     ${SAX_CELL}/${MODEL_NAME} \
     ${MODEL_CONFIG_PATH} \
     ${CHECKPOINT_PATH} \
     ${REPLICA}
```

- Unpublish model
```bash
docker run \
 ${SAX_UTIL_IMAGE_URL} \
   --sax_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
   unpublish \
     ${SAX_CELL}/${MODEL_NAME}
```

- List models
```bash
docker run \
 ${SAX_UTIL_IMAGE_URL} \
   --sax_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
   ls ${SAX_CELL}/${MODEL_NAME}
```

### Test inference

```bash
docker run \
  ${SAX_UTIL_IMAGE_URL} \
    --sax_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
    lm.generate \
      ${SAX_CELL}/${MODEL_NAME} \
      ${INPUT_STR}
```

## Appendix

- Run SAX util
```bash
docker run \
    --privileged  \
    -it \
    --rm \
    --name sax_util \
    -v /mnt/disks/saxml:/saxml_inf \
    --entrypoint /bin/bash \
    ${SAX_UTIL_IMAGE_URL}
```

- TPU v4 start/stop
```bash
curl -X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
-d "{metadata: {'enable-oslogin': 'FALSE'}}" https://tpu.googleapis.com/v2/projects/$PROJECT_ID/locations/$ZONE/nodes/$TPU_NAME?updateMask=metadata


curl -s -k -X POST -H "Content-Type: application/json" \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-d "{}" https://tpu.googleapis.com/v2/projects/${PROJECT_ID}/locations/${ZONE}/nodes/${TPU_NAME}:start

curl -s -k -X POST -H "Content-Type: application/json" \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-d "{}" https://tpu.googleapis.com/v2/projects/${PROJECT_ID}/locations/${ZONE}/nodes/${TPU_NAME}:stop

curl -H "Content-Type: application/json" \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
https://tpu.googleapis.com/v2/projects/${PROJECT_ID}/locations/${ZONE}/nodes
```

- Attach disk to TPU
```bash
gcloud compute disks create dsk-sax-tpu \
    --size 500GB  \
    --zone ${ZONE} \
    --type pd-balanced

gcloud alpha compute tpus tpu-vm attach-disk ${TPU_NAME} \
    --zone=${ZONE} \
    --disk=dsk-sax-tpu \
    --mode=read-write
```

- Generate INPUT_STR for inference with
```python
TEXT = """Below is an instruction that describes a task, paired with
an input that provides further context. Write a response that
appropriately completes the request.\n\n### Instruction\:\nSummarize the
following news article\:\n\n### Input\:\nMarch 10, 2015 . We're truly
international in scope on Tuesday. We're visiting Italy, Russia, the
United Arab Emirates, and the Himalayan Mountains. Find out who's
attempting to circumnavigate the globe in a plane powered partially by the
sun, and explore the mysterious appearance of craters in northern Asia.
You'll also get a view of Mount Everest that was previously reserved for
climbers. On this page you will find today's show Transcript and a place
for you to request to be on the CNN Student News Roll Call. TRANSCRIPT .
Click here to access the transcript of today's CNN Student News program.
Please note that there may be a delay between the time when the video is
available and when the transcript is published. CNN Student News is
created by a team of journalists who consider the Common Core State
Standards, national standards in different subject areas, and state
standards when producing the show. ROLL CALL . For a chance to be
mentioned on the next CNN Student News, comment on the bottom of this page
with your school name, mascot, city and state. We will be selecting
schools from the comments of the previous show. You must be a teacher or a
student age 13 or older to request a mention on the CNN Student News Roll
Call! Thank you for using CNN Student News!\n\n### Response\:"""

from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/saxml_inf/llama2/hf/llama-2-7b")
model = LlamaForCausalLM.from_pretrained("/saxml_inf/llama2/hf/llama-2-7b")


import torch
from transformers import AutoTokenizer, GPTJForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
model = GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")

from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
gen(TEXT)
```