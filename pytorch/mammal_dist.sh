#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=2
fi

echo "Using $NTHREADS threads"

# make sure OpenMP doesn't interfere with pytorch.multiprocessing
#export OMP_NUM_THREADS=1

mpirun -np 2 \
    -H localhost:2 \
    python embed_dist.py \
    -dim 5 \
    -lr 0.1 \
    -epochs 600 \
    -negs 50 \
    -burnin 20 \
    -nproc "${NTHREADS}" \
    -distfn poincare \
    -dset wordnet/mammal_closure.tsv \
    -fout mammal.pth \
    -batchsize 10 \
    -eval_each 1 \


#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl ^openib \
