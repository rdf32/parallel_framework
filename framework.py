import inspect
from tqdm import tqdm
from functools import wraps

from mpi4py import MPI

mpi_world = MPI.COMM_WORLD
mpi_world.barrier() # synchronization

def user_defined(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(**(dict(zip(list(inspect.signature(func).parameters.keys())[:len(args)], args)) 
                       | {key:kwargs[key] for key in list(inspect.signature(func).parameters.keys()) if key in kwargs}))
    return wrapper


class SerialProcess:
    def __init__(self, chunkfunc, workfunc, resultfunc, initfunc=None, finalfunc=None, **kwargs):
        self.cfunc = chunkfunc
        self.wfunc = workfunc
        self.rfunc = resultfunc
        self.initfunc = initfunc
        self.ffunc = finalfunc
        self.kwargs = kwargs

    def __call__(self):
        print("serial processing begins...")
        self.master(**self.kwargs)
        return self.final(**self.kwargs)
    
    def master(self, **kwargs):
        kwargs.update(self.ifunc(**kwargs)) if self.ifunc is not None else print ('no intitialization required') ## updates kwargs with output of init func -> return a dictionary
        total_chunks, chunks = self.cfunc(**kwargs)

        for _ in tqdm(range(total_chunks)):
            result, chunk = self.worker(next(chunks, None), **kwargs)
            _ = self.rfunc(result, **({'chunk': chunk} | kwargs))

    def worker(self, chunk, **kwargs):
        if chunk is not None:
            results = self.wfunc(chunk, **kwargs)
            return results, chunk
        
    def final(self, **kwargs):
        self.ffunc(**kwargs) if self.ffunc is not None else None

class ParrallelProcess:
    def __init__(self, chunkfunc, workfunc, resultfunc, initfunc=None, finalfunc=None, **kwargs):
        self.cfunc = chunkfunc
        self.wfunc = workfunc
        self.rfunc = resultfunc
        self.initfunc = initfunc
        self.ffunc = finalfunc
        self.kwargs = kwargs

    def __call__(self):
        self.master(mpi_world, **self.kwargs) if (mpi_world.rank == 0) else self.worker(mpi_world, **self.kwargs)
        mpi_world.barrier()
        results = self.final(**self.kwargs) if (mpi_world.rank == 0) else None
        mpi_world.barrier()
        return results
    

    def master(self, mpi_world, **kwargs):
        print('parrallel processing begins....')
        print(f'Number of Workers: {mpi_world.size}')

        mpi_status = MPI.status()
        kwargs.update(self.ifunc(**kwargs)) if self.ifunc is not None else print ('no initialization required')

        total_chunks, chunks = self.cfunc(**kwargs)

        # give out starting chunks to each worker
        for worker in range(1, mpi_world.size):
            mpi_world.send(next(chunks, None), dest=worker)

        # listen for results and handle them then send more work
        for _ in tqdm(range(total_chunks)):
            result, chunk = mpi_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=mpi_status)
            _ = self.rfunc(result, **({'chunk': chunk} | kwargs))
            # send a new chunk if there are any remaining
            mpi_world.send(next(chunks, None), dest=mpi_status.Get_source())
        
    def worker(self, mpi_world, **kwargs):
        chunk = mpi_world.recv(source=0)
        while chunk is not None:
            results = self.wfunc(chunk, **kwargs)
            mpi_world.send((results, chunk), dest=0)
