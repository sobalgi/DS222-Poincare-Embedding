

TensorFlow implementation of [[https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations][Poincaré Embeddings for Learning Hierarchical Representations]]

[[file:wn-nouns.jpg]]

** Example: Embedding WordNet Mammals
To embed the transitive closure of the WordNet mammals subtree, first generate the data via
#+BEGIN_SRC sh
  cd wordnet
  make
#+END_SRC
This will generate the transitive closure of the full noun hierarchy as well as of the mammals subtree of WordNet. 

To embed the mammals subtree in the reconstruction setting (i.e., without missing data), go to the /root directory/ of the project and run
#+BEGIN_SRC sh
  NTHREADS=2 ./train-mammals.sh
#+END_SRC
This shell script includes the appropriate parameter settings for the mammals subtree and saves the trained model as =mammals.pth=. 

An identical script to learn embeddings of the entire noun hierarchy is located at =train-nouns.sh=. This script contains the hyperparameter setting to reproduce the results for 10-dimensional embeddings of [[https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations][(Nickel & Kiela, 2017)]]. The hyperparameter setting to reproduce the MAP results are provided as comments in the script.

The embeddings are trained via multithreaded async SGD. In the example above, the number of threads is set to a conservative setting (=NHTREADS=2=) which should run well even on smaller machines. On machines with many cores, increase =NTHREADS= for faster convergence.

** Dependencies
- Python 3 with NumPy
- TensorFlow
- Scikit-Learn
- NLTK (to generate the WordNet data)
