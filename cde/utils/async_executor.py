from multiprocessing import Process
import time

class AsyncExecutor:

    def __init__(self, num_workers=1):
        self.num_workers = num_workers
        self._pool = []
        self._populate_pool()

    def run(self, target, *args_iter):
        workers_idle = [False] * self.num_workers
        tasks = list(zip(*args_iter))

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not self._pool[i].is_alive():
                    self._pool[i].terminate()
                    if len(tasks) > 0:
                        next_task = tasks.pop(0)
                        self._pool[i] = start_process(target, next_task)
                    else:
                        workers_idle[i] = True

    def _populate_pool(self):
        self._pool = [start_process(_dummy_fun) for _ in range(self.num_workers)]




def start_process(target, args=None):
    if args:
        p = Process(target=target, args=args)
    else:
        p = Process(target=target)
    p.start()
    return p

def _dummy_fun():
    pass

# def foo(a,b):
#     time.sleep(5)
#     print(a,b)
#
#
# if __name__ == "__main__":
#     exec = AsyncExecutor(num_workers=10)
#     exec.run(foo, range(20), range(20))

