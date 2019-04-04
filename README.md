# TextGAN

Project of course "Structured Data: Learning, Prediction Dependency, Testing" of Master Data Sciences (École Polytechnique).

## Prerequisite

PyTorch v1.0.1

## Data Link

[Three_corpus](https://drive.google.com/open?id=0B52eYWrYWqIpd2o2T1E3aUU0cEk)

## Citation

This project is an implementation of the framework TextGAN proposed in the following papers:

* **Adversarial Feature Matching for Text Generation**,
Yizhe Zhang, Zhe Gan, Kai Fan, Zhi Chen, Ricardo Henao, Lawrence Carin. ICML, 2017.
* **Generating Text via Adversarial Training.**
Yizhe Zhang, Zhe Gan, Lawrence Carin.  Workshop on Adversarial Training, NIPS, 2016.

## Structure

I implemented mainly the basic architecture of TextGAN. The pre-training and approximation techniques used by authors are not yet finished.

`textGAN.py` gives the architecture

`train.py` can be run to train a TextGAN model (TO FINISH)

`utils.py` gives some helper functions
