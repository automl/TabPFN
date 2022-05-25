import os
from pathlib import Path

from samlib import submitit
from samlib.utils import chunker
from typing import List
from submitit import SlurmExecutor
import io
import torch
import pickle

def print_models(base_path, model_string):
    print(model_string)

    for i in range(80):
        for e in range(50):
            exists = Path(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt')).is_file()
            if exists:
                print(os.path.join(base_path, f'models_diff/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.cpkt'))
        print()


def set_queue(q_):
    global ex
    global log_folder

    if q_ == 'bosch':
        print('bosch')

        class BoschSlurmExecutor(SlurmExecutor):
            def _make_submission_command(self, submission_file_path) -> List[str]:
                return ["sbatch", str(submission_file_path), '--bosch']

        ex = BoschSlurmExecutor(folder=log_folder)
        ex.update_parameters(time=1200
                             , partition="bosch_cpu-cascadelake"
                             , mem_per_cpu=6000
                             , nodes=1
                             , cpus_per_task=1
                             , ntasks_per_node=1
                             #                     , setup=['export MKL_THREADING_LAYER=GNU']
                             )  # mldlc_gpu-rtx2080

        return 1200

    if q_ == 'all':
        q = 'alldlc_gpu-rtx2080'
    if q_ == 'ml':
        q = 'mldlc_gpu-rtx2080'

    if q == 'alldlc_gpu-rtx2080':
        maximum_runtime = 24 * 60 * 1 - 1
    else:
        maximum_runtime = 24 * 60 * 4 - 1

    ex = submitit.AutoExecutor(folder=log_folder)
    ex.update_parameters(timeout_min=maximum_runtime, slurm_partition=q)  # mldlc_gpu-rtx2080

    return maximum_runtime


def cancel_all_jobs(job_queue):
    for k in job_queue.keys():
        for j in job_queue[k]:
            j.cancel()


def cancel_job(k):
    global job_queue
    for j in job_queue[k]:
        j.cancel()

def cancel_last_jobs():
    global job_queue
    for j in job_queue[list(job_queue.keys())[-1]]:
        j.cancel()

def check_last_jobs_done():
    global job_queue
    for j in job_queue[list(job_queue.keys())[-1]]:
        print(j.done())


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        try:
            return self.find_class_cpu(module, name)
        except:
            return None

    def find_class_cpu(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)