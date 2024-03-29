Hierarchical Dense Subtensor Detection in Tensors
========

[![Build Status](https://travis-ci.com/wenchieh/catchcore.svg?branch=master)](https://travis-ci.com/wenchieh/catchcore) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) ![GitHub](https://img.shields.io/github/license/wenchieh/catchcore.svg)


**CatchCore** is a novel framework to detect hierarchical dense cores in multi-aspect data (i.e. tensors).
CatchCore has the following properties:

- unified metric: provides a gradient-based optimized framework as well as theoretical guarantees
- accurate: provides high accuracy in both synthetic and real data
- effectiveness: spots anomaly patterns and hierarchical dense community
- scalable: scales almost linearly with all factors of input tensor, also has linearly space complexity

----


Datasets
========================

The download links for the datasets used in the paper are available online.
  - [Android App rating](http://jmcauley.ucsd.edu/data/amazon/).  1.32M × 61.3K × 1.28K × 5
  - [BeerAdvocate rating](http://snap.stanford.edu/data/web-BeerAdvocate.html).  26.5K × 50.8K × 1472 × 1
  - [StackOverflow favorite](http://konect.cc/networks/stackexchange-stackoverflow/).  545K × 96.7K × 1.15K × 1
  - [DBLP Co-author](https://networkrepository.com/dblp_coauthor.php).   1.31M × 1.31M × 72
  - [Youtube Favorite](http://konect.cc/networks/youtube-u-growth/).   3.22M × 3.22M × 203
  - [DARPA TCP Dumps](https://www.ll.mit.edu/r-d/datasets/1999-darpa-intrusion-detection-evaluation-dataset).   9.48K × 23.4K × 46.6K
  - [AirForce TCP Dumps](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).  3 × 70 × 11 × 7.20K × 21.5K × 512 × 512


Environment
=======================
To install required libraries, please type
```bash
pip install -r requirements
```
----

Building and Running CatchCore
========================
Please see [User Guide](user_guide.pdf)

---

Running Demo
========================

Demo for detecting hierarchical dense subtensor, please type
```bash
make
```


Reference
========================
If you use this code as part of any published research, please acknowledge the following papers.
```
@article{feng2023hierarchical,
  title={Hierarchical Dense Pattern Detection in Tensors},
  author={Feng, Wenjie and Liu, Shenghua and Cheng, Xueqi},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={17},
  number={6},
  pages={1--29},
  year={2023},
  publisher={ACM New York, NY}
}

@inproceedings{feng2019catchcore,
  title={CatchCore: Catching Hierarchical Dense Subtensor},
  author={Wenjie Feng, Shenghua Liu, and Xueqi Cheng},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year={2019},
}

```
