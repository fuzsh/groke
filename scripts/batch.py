import json
import time
from typing import Optional, List

from google import genai
from google.genai import types
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from pydantic import BaseModel

client = genai.Client(http_options=HttpOptions(api_version="v1"), vertexai=True, project='gen-lang-client-0457627383', location='global')

step_number = 8
ttype = 'unseen'

input_file_src = f"gs://sisu-aalto/main_prompt_{ttype}/step{step_number}_json.jsonl"
output_uri = f"gs://sisu-aalto/main_{ttype}"


job = client.batches.create(
    model="gemini-3-pro-preview",
    src=input_file_src,
    config=CreateBatchJobConfig(dest=output_uri),
)

print(f"Job name: {job.name}")
print(f"Job state: {job.state}")
# Example response:
# Job name: projects/.../locations/.../batchPredictionJobs/9876453210000000000
# Job state: JOB_STATE_PENDING

# See the documentation: https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob
completed_states = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}

print(f"Job name: {job.state}")


while job.state not in completed_states:
    time.sleep(15)
    job = client.batches.get(name=job.name)
    print(f"Job state: {job.state}")
# Example response:
# Job state: JOB_STATE_PENDING
# Job state: JOB_STATE_RUNNING
# Job state: JOB_STATE_RUNNING
# ...
# Job state: JOB_STATE_SUCCEEDED
