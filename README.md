# OpsLens

![Python Version](https://img.shields.io/badge/Python-3776AB?&logo=python&logoColor=white-blue&label=3.12)&ensp;

We propose OpsLens, a generalizable RCA solution that performs collaborative analysis on multi-modal observability data. OpsLens eliminates the need for prebuilt tools or historical incident reports. It uses an LLM-based multi-agent system (MAS) and the two-stage tool generation with feedback-based refinement to better interact with observability data and perform collaborative analysis. 

Experiments on public datasets  from three real-world systems  containing 335  incidents show that OpsLens achieves state-of-the-art RCA performance, with a score rate of 0.80.

## ✨ Features
- LLM-based multi-agent system (MAS)
- Two-stage tool generation with feedback-based refinement
- Better interact with observability data and perform collaborative analysis. 

## 💡Prerequisites
OpsLens requires **Python >= 3.12**. 
```
# create conda env
conda create -n opslens python==3.12
conda activate opslens
pip install -r requirements.txt
```
### docker
To ensure environmental safety, we run the generated code in Docker. Therefore, please make sure that Docker is installed on your machine before running the code.
Run the following command to check if the Docker daemon is running:
```
docker version
```
If the terminal outputs `Cannot connect to the Docker daemon at xxx Is the docker daemon running?`, please check the running status of Docker again.

For Linux machines, please refer to: [Linux | Docker Docs](https://docs.docker.com/desktop/setup/install/linux/)
For Mac machines, in addition to Docker Desktop, you can also use Orbstack as an alternative:
```
brew install orbstack
```
### dataset
In addition to the environment, we use OpenRCA as the dataset. You can download the data from [Google Drive](https://drive.google.com/drive/folders/1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM) and then place it in the `coding/dataset` directory under the coding file set.
```
├── coding
│   └── dataset
│       └── Bank
│           └── telemetry
│               ├── 2021_03_04
│               │   ├── log
│               │   │   └── log_service.csv
│               │   ├── metric
│               │   │   ├── metric_app.csv
│               │   │   └── metric_container.csv
│               │   └── trace
│               │       └── trace_span.csv
│               └── 2021_03_05
│                   ├── log
│                   │   └── log_service.csv
│                   ├── metric
│                   │   ├── metric_app.csv
│                   │   └── metric_container.csv
│                   └── trace
│                       └── trace_span.csv
...
...

```
## 🛠️ How to Run
First, you need to add your api_key in `agent.py`.
```python
reason_model_client =  OpenAIChatCompletionClient(
        model="",
        base_url="",
        api_key="",
        model_info={
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        },
        temperature=0,
)

model_client =  OpenAIChatCompletionClient(
        model="",
        base_url="",
        api_key="",
        model_info={
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        },
        temperature=0,
)
```
Then,  you can run OpsLens to perform RCA on a given query by running:
```bash
python opslens.py --query "On March 4, 2021, between 18:00 and 18:30, there was a single failure observed in the system. The exact component that caused this failure is unknown, and the reason behind the failure is also undetermined. Your task is to identify the root cause component and the root cause reason for this failure."
```

## 📊 How to Evaluate
We evaluate OpsLens on three real-world systems: Bank, Market, and Telecom. 
You can reproduce the evaluation results by running:
```bash
python -m eval.evaluate \
    -p \
      ./archive/opslens-eval-bank.csv \
      ./archive/opslens-eval-telecom.csv \
      ./archive/opslens-eval-market1.csv \
      ./archive/opslens-eval-market2.csv \
    -q \
      ./query/bank_query.csv \
      ./query/telecom_query.csv \
      ./query/market1_query.csv \
      ./query/market2_query.csv \
    -r \
      ./report.csv
```

## 📒Notes
Since autogen now deletes the container every time it finishes running, if you don't want to re-initialize the environment every time, you can modify the `start` function in the following file to specify the container.
`YOURPATH/miniconda3/envs/autogen/lib/python3.12/site-packages/autogen_ext/code_executors/docker/_docker_code_executor.py`

```python
async def start(self) -> None:
    """(Experimental) Start the code executor.

    This method sets the working environment variables, connects to Docker and starts the code executor.
    If no working directory was provided to the code executor, it creates a temporary directory and sets it as the code executor working directory.
    """

    if self._work_dir is None and self._temp_dir is None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._temp_dir_path = Path(self._temp_dir.name)
        self._temp_dir_path.mkdir(exist_ok=True)

    # Start a container from the image, read to exec commands later
    try:
        client = docker.from_env()
    except DockerException as e:
        if "FileNotFoundError" in str(e):
            raise RuntimeError("Failed to connect to Docker. Please ensure Docker is installed and running.") from e
        raise
    except Exception as e:
        raise RuntimeError(f"Unexpected error while connecting to Docker: {str(e)}") from e
    
    #------------------------NEW CODE-------------------
    # try to get the container if it exists
    try:
        container = await asyncio.to_thread(client.containers.get, self.container_name)
        if container.status != "running":
            await asyncio.to_thread(container.start)
        self._container = container
        self._running = True
        await _wait_for_ready(self._container)
        return
    except NotFound:
        pass  # Container does not exist, continue to create a new one
    #------------------------NEW CODE-------------------

    # Check if the image exists
    try:
        await asyncio.to_thread(client.images.get, self._image)
    except ImageNotFound:
        # TODO logger
        logging.info(f"Pulling image {self._image}...")
        # Let the docker exception escape if this fails.
        await asyncio.to_thread(client.images.pull, self._image)

    # Prepare the command (if needed)
    shell_command = "/bin/sh"
    command = ["-c", f"{(self._init_command)};exec {shell_command}"] if self._init_command else None

    # Check if a container with the same name already exists and remove it
    try:
        existing_container = await asyncio.to_thread(client.containers.get, self.container_name)
        await asyncio.to_thread(existing_container.remove, force=True)
    except NotFound:
        pass

    self._container = await asyncio.to_thread(
        client.containers.create,
        self._image,
        name=self.container_name,
        entrypoint=shell_command,
        command=command,
        tty=True,
        detach=True,
        auto_remove=self._auto_remove,
        volumes={str(self.bind_dir.resolve()): {"bind": "/workspace", "mode": "rw"}, **self._extra_volumes},
        working_dir="/workspace",
        extra_hosts=self._extra_hosts,
        device_requests=self._device_requests,
    )
    await asyncio.to_thread(self._container.start)

    await _wait_for_ready(self._container)

    async def cleanup() -> None:
        await self.stop()
        asyncio_atexit.unregister(cleanup)  # type: ignore

    if self._stop_container:
        asyncio_atexit.register(cleanup)  # type: ignore

    # Check if the container is running
    if self._container.status != "running":
        logs_str = self._container.logs().decode("utf-8")
        raise ValueError(f"Failed to start container from image {self._image}. Logs: {logs_str}")

    self._loop = asyncio.get_running_loop()
    self._cancellation_futures = []
    logging.debug(f"Executor started, associated with event loop: {self._loop!r}")

    self._running = True
```    
