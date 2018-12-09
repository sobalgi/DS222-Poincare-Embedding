

TensorFlow implementation of [[https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations][Poincaré Embeddings for Learning Hierarchical Representations]]

[[file:wn-nouns.jpg]]

## Example: Embedding WordNet Mammals
To embed the transitive closure of the WordNet mammals subtree, first generate the data via
```
  cd wordnet
  make
```
This will generate the transitive closure of the full noun hierarchy as well as of the mammals subtree of WordNet. 

The embeddings are trained via Asynchronous Gradient Descent Optimizer for the distributed code. In the example above, the number of workers were set to 1 which should run well even on a single machine. On machines with many cores, increase workers for faster convergence.

### for local code

```
cd ds222/assignment2   
python poincare_tensor.py
```

### for distributed code (change the nodes accordingly)

```
cd ds222/assignment2  
pc-01$ python poincare_async.py --job_name="ps" --task_index=0     
pc-02$ python poincare_async.py --job_name="worker" --task_index=0     
pc-03$ python poincare_async.py --job_name="worker" --task_index=1     
pc-04$ python poincare_async.py --job_name="worker" --task_index=2    
```
Similarly for Drop Stale Synchronous (poincare_stale.py)

### Dependencies
- Python 3 with NumPy
- TensorFlow
- Scikit-Learn
- NLTK (to generate the WordNet data)
- HOROVOD (for the HOROVOD code)

### References
[Distibuted tensorflow example] (https://github.com/ischlag/distributed-tensorflow-example)  
[Distributed tensorflow documentation] (http://www.tensorflow.org/deploy/distributed)
