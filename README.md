# Quantum Kernel Mixtures

Quantum kernel mixtures provide a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. The framework allows for the construction of differentiable models for density estimation, inference, and sampling, enabling integration into end-to-end deep neural models. 

## Paper

> **Quantum Kernel Mixtures for Probabilistic Deep Learning**
> 
> Fabio A. González, Raúl Ramos-Pollán, Joseph A. Gallego-Mejia
> 
> https://arxiv.org/abs/2305.18204
> 
> <p align="justify"><b>Abstract:</b> <i>This paper presents a novel approach to probabilistic deep learning (PDL), quantum kernel mixtures, derived from the mathematical formalism of quantum density matrices, which provides a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. The framework allows for the construction of differentiable models for density estimation, inference, and sampling, enabling integration into end-to-end deep neural models. In doing so, we provide a versatile representation of marginal and joint probability distributions that allows us to develop a differentiable, compositional, and reversible inference procedure that covers a wide range of machine learning tasks, including density estimation, discriminative learning, and generative modeling. We illustrate the broad applicability of the framework with two examples: an image classification model, which can be naturally transformed into a conditional generative model thanks to the reversibility of our inference procedure; and a model for learning with label proportions, which is a weakly supervised classification task, demonstrating the framework's ability to deal with uncertainty in the training samples.</i></p>

## Citation

If you find this code useful in your research, please consider citing:

```
@misc{gonzalez2023quantum,
      title={Quantum Kernel Mixtures for Probabilistic Deep Learning}, 
      author={Fabio A. González and Raúl Ramos-Pollán and Joseph A. Gallego-Mejia},
      year={2023},
      eprint={2305.18204},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
