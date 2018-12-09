#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=2
fi

echo "Using $NTHREADS threads"

# make sure OpenMP doesn't interfere with pytorch.multiprocessing
export OMP_NUM_THREADS=1

python3 embed.py \
       -dim 5 \
       -lr 0.3 \
       -epochs 3000 \
       -negs 50 \
       -burnin 20 \
       -nproc ${NTHREADS} \
       -ndproc 2
       -distfn poincare \
       -dset wordnet/noun_closure.tsv \
       -fout noun.pth \
       -batchsize 100 \
       -eval_each 10 \