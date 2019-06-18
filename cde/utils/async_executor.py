from multiprocessing import Process
from multiprocessing import Manager
import multiprocessing
import numpy as np
from progressbar.bar import ProgressBar

class AsyncExecutor:

    def __init__(self, n_jobs=1):
        self.num_workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter, verbose=False):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        if verbose:
          pbar = ProgressBar(max_value=n_tasks)

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        if verbose:
                          pbar.update(n_tasks-len(tasks))
                          #print(n_tasks-len(tasks))
                        next_task = tasks.pop(0)
                        self._pool[i] = _start_process(target, next_task)
                    else:
                        workers_idle[i] = True
        if verbose:
            pbar.finish()

    def _populate_pool(self):
        self._pool = [_start_process(_dummy_fun) for _ in range(self.num_workers)]

class LoopExecutor:

    def run(self, target, *args_iter, verbose=False):
        tasks = list(zip(*args_iter))
        n_tasks = len(tasks)

        if verbose:
          pbar = ProgressBar(max_value=n_tasks)

        for i, task in enumerate(tasks):
            target(*task)
            if verbose:
                pbar.update(i+1)


def execute_batch_async_pdf(pdf_fun, X, Y, n_jobs=-1, batch_size=None):
    """
    Executes pdf_fun in batches in multiple processes and concatenates results along axis 0

    Args:
        pdf_fun: callable with signature pdf(X, Y) returning a numpy array
        X: ndarray with shape (n_queries, ndim_x)
        Y: ndarray with shape (n_queries, ndim_y)
        n_jobs: integer denoting the number of jobs to launch in parallel. If -1
                      it uses the CPU count
        batch_size: (optional) integer denoting the batch size for the individual function calls

    Returns:
        ndarray of shape (n_queries,) which results from a concatenation of all pdf calls
    """
    # split query arrays into batches
    query_length = X.shape[0]

    if n_jobs < 1:
        n_jobs = max(multiprocessing.cpu_count(), 8)

    if batch_size is None:
        n_batches = n_jobs
    else:
        n_batches = query_length // batch_size + int(not (query_length % batch_size == 0))

    X_batches, Y_batches, indices = _split_into_batches(X, Y, n_batches)


    # prepare multiprocessing setup
    manager = Manager()
    result_dict = manager.dict()

    def run_pdf_async(X_batch, Y_batch, batch_idx):
        p = pdf_fun(X_batch, Y_batch)
        result_dict[batch_idx] = p

    # compute pdf for batches asynchronously
    executer = AsyncExecutor(n_jobs=n_jobs)
    executer.run(run_pdf_async, X_batches, Y_batches, indices)

    # concatenate results
    p_final = np.concatenate([result_dict[i] for i in indices], axis=0)
    assert p_final.shape[0] == query_length
    return p_final


def _split_into_batches(X, Y, n_batches):
    assert X.shape[0] == X.shape[0]
    if n_batches <= 1:
        return [X], [Y], range(1)
    else:
        return np.array_split(X, n_batches, axis=0), np.array_split(Y, n_batches, axis=0), range(n_batches)




""" helpers """

def _start_process(target, args=None):
    if args:
        p = Process(target=target, args=args)
    else:
        p = Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass


