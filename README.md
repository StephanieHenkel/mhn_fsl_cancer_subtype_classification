# Modern Hopfield Networks for Few-and Zero-Shot Cancer Subtype Classification

This repository contains the corresponding code for my Master Thesis
with the title ["Modern Hopfield Networks for Few-and Zero-Shot 
Cancer Subtype Classification"](https://epub.jku.at/obvulihs/download/pdf/8080454?originalFilename=true "Master Thesis").


The code uploaded to this repository is only a part of the code and work
I have done for this Thesis. However, I consider these files as the most
important ones and think that other files are not interesting or helpful
to understand this work.

The two main files are:
- [create_pretrained_networks.py](code/create_pretrained_networks.py)
  - creates pretrained models for a specific network type 
    and pooling function
- [run_nested_cv.py](code/run_nested_cv.py)
  - runs nested cross-validation over priorly chosen pretrained 
    models and a set of hyperparameters to receive the best 
    possible fine-tuned models

The network types are defined in [Models.py](code/Models.py)
and the pooling functions are defined in 
[Pooling_Functions.py](code/Pooling_Functions.py).
