

## Example: Embedding WordNet Mammals
To embed the transitive closure of the WordNet mammals subtree, first generate the data via
```
  cd pytorch/wordnet
  make
```
This will generate the transitive closure of the full noun hierarchy as well as of the mammals subtree of WordNet. 

The embeddings are trained via Asynchronous Gradient Descent Optimizer for the distributed code. In the example above, the number of workers were set to 1 which should run well even on a single machine. On machines with many cores, increase workers for faster convergence.

### Adjust the filenames (mammals_closure.tsv or noun_closure.tsv) 

 * In utils.py
```
15 targets = set(open('/data/targets.txt').read().split('\n'))
23 with open('/data/mammal_closure.tsv', 'w') as out:
84 dp='/data/mammal_closure.tsv')
```
 * In poincare_tensor.py
```
 def __init__(self,num_iter=100,num_negs=10,lr1=0.001,lr2=0.00001,dp='/home/rishixtreme/PycharmProjects/poincare/data/mammal_closure.tsv')
```

### Running the local code

```
cd ds222/assignment2   
python poincare_tensor.py
```

### Running the distributed code (change the nodes accordingly)

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
[Distibuted tensorflow example](https://github.com/ischlag/distributed-tensorflow-example)  
[Distributed tensorflow documentation](http://www.tensorflow.org/deploy/distributed)
