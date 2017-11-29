### Code for the paper "New Algorithms for Inference in Graph Sequence Models" (2017)

This codebase is for algorithms proposed in the aforementioned [paper](https://theja.org/#). It consists of the following python scripts:

  * graph_generators.py : has the synthetic/real instance generators
  * graph_estimators.py : has the proposed algorithms
  * experiments.py : calls algorithms on various instances
  * plots_paper.py : generates plots given in the paper

The easiest way to get started is to look at experiments.py and go from there. 

##### Dependencies

These are: numpy, networkx, [graph-tool](https://graph-tool.skewed.de/)

Please make a pull request if you spot bugs or have suggestions!
Last model will change.
