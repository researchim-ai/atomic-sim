"""MPI runtime transport probe only; no claim of an MPI reactor solver.

API reference: https://mpi4py.readthedocs.io/en/stable/tutorial.html
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size < 2:
        raise ValueError("MPI probe requires at least 2 ranks")
    values = comm.gather((rank + 1) ** 2, root=0)
    if rank == 0:
        from atomic_model_workspace.common import atomic_json

        expected = [(i + 1) ** 2 for i in range(size)]
        if values != expected:
            raise ValueError("MPI gathered values differ")
        atomic_json(
            args.output,
            dict(
                schema_version=1,
                ranks=size,
                transport_passed=True,
                values=values,
                reactor_solver_qualified=False,
                validated=False,
                plant_referenced=False,
            ),
        )
    comm.Barrier()


if __name__ == "__main__":
    main()
