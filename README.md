# Local Low Latency Speech to Speech

This repository provides a solution for low-latency speech-to-speech conversion, catering to users with and without GPUs for accelerated processing. Follow the steps below to set up and utilize the system.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/All-About-AI-YouTube/low-latency-sts.git
    ```

2. Navigate to the cloned directory:
    ```bash
    cd low-latency-sts
    ```

3. Set up the environment:
    ```bash
    conda create -n openvoice python=3.9
    conda activate openvoice
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```

4. Create a folder named "checkpoints".

5. Download the checkpoints from [this link](https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip).

6. Unzip the downloaded file and move its contents (basespeakers + converter) to the "checkpoints" folder.

7. Install LM Studio from [here](https://lmstudio.ai/).

8. Download the Bloke Dolphin Mistral 7B V2 model from [Hugging Face](https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-AWQ) and import it into LM Studio.

9. Set up a Local Server in LM Studio following the instructions [here](https://youtu.be/IgcBuXFE6QE).

10. Start the server.

## Usage Instructions

- For users with GPUs:
    - Run `voice69.py` followed by `talk.py`.

- For users without GPUs, seeking faster response:
    - Utilize `talk_with_bark.py`.

Ensure that you have the necessary reference voice (in mp3 format) in the specified PATH or PATHS before running the scripts.



