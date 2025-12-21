# Transformer Algebra

This project is intended to provide a tool to support both interactive and automated analysis of the internal states of transformer based large language models, keeping a close link between code and notation. It should be able to answer questions like why the algorithm generate a specific output, and to provide labels for internal states and subsets of the model weights which are easier to interpret.

# Language Model Interpretability

We have  a very poor understanding of how LLMs are able to do what they do. They are a collection of billions of weights which when multiplied together in the right order bring back a meaningful result. We can analyse model internals to find particular weights relating to specific concepts or knowledge but this requires advanced mathematical knowledge. Tools such as Transformer Lens can analyse attention weights to indicate which tokens the model has referred to in generating specific outputs, but without further analysis this only suggests the logic of specific outputs. Logit Lens and Tuned Lens analyse the internal state through a model with reference to the next generated tokens. Research tools have been developed to modify weights to erase specific concepts, but the most effective way to interact with the data held in the model is using the model, providing instructions and asking questions using natural language. 

# Approach

Transformer Algebra aims to define a a concise mathematical syntax which emphasises the vector operations performed, and implement an interpreter which uses this notation to perform inference using a specific model, and allow expansion of terms to show intermediate steps.

The core is to think of a combination of a specific instantiated transformer model with context text as a single operator or function acting on the embedding vector for the next token to transform to an unembedding vector which determines a probability function for the next token.
We aim to hide as much detail as possible focussing on the embedding/unembedding text and associated vectors, not on the model and weights. Where possible we will represent operations based on the angle between named vectors or subspaces rather than tensor operations, but remain grounded in actual operations performed by a specific model in a specific context.

# Tasks

## Define notation

See notebooks\Summary.ipynb for initial suggestions. The notation should be concise and unambiguous. Fundamental elements include naming elements of the embedding and unembedding vectors based on their tokens, and defining notation to describe the transformer as a whole, and blocks, layers heads etc within.

## Implement interpreter using small models

Use pythia models due to the availability of very small models and snapshots during training.

The basis of the interpreter is to treat the combination of model and context as a single operator acting on the embedding vector for the last token to produce an unembedding vector. With this working the notation can be expanded to cover multiple tokens as input and output.

One representation is the unembedding vector interpreted as a probability distribution over the vocabulary for the next token. 

## Implement expansion of terms

The basic operation of the transformer (combination of model and context) can be expanded to a sequence of operations for each block of the model. With possible approximation this can be represented as a linear combination of operations, each of which can be broken down further to show attention heads and MLPs. This process of expansion should be interactive initially, though may later be automated to identify the most significant components.

Mathematical reasoning may be required to identify useful expansions and approximations.

## Implement naming of vectors and subspaces

As mentioned, embedding and unembedding vectors can be named right away. The language model itself may be of use in naming specific relevant vectors or subspaces within the model weights. This may be done interactively at first, with a view to automating later.

## Validate against experiments

Throughout the development process, the results should be validated against experiments using standard model implementations.

# Long term Goals

- Support mechanistic interpretability research.
- Show whether a specific generated tokens are more influenced by part of the context, or by weights in the underlying model.
- Provide a way to label and name specific vectors or subspaces within the model weights to support interpretability.
- Support better use of Retrieval-Augmented Generation

# Current Status

`notebooks/investigate.ipynb` demonstrates basic logit lens analysis, showing numeric logit values for a target token across layers. The next phase transforms this into **symbolic output** with named vectors and operators.

## Next Steps

- Implement contribution analysis for expanded terms
- Extract per-block contributions
- Export to Julia via HDF5

See `doc/roadmap.md` for detailed planning and `doc/notation.md` for the mathematical notation.

## Related Repositories

- **[SymbolicTransformer](https://github.com/prior-technology/SymbolicTransformer)** - Julia symbolic manipulation and notation rendering
- **[VectorTransformer](https://github.com/prior-technology/VectorTransformer)** - Julia reference implementation for validation

# History

Key influences are "A Mathematical Framework for Transformer Circuits" and "logit lens".

An earlier implementation [SymbolicTransformer](https://github.com/prior-technology/SymbolicTransformer) aimed to use features of Julia towards the same goal in particular support for mathematical based syntax and macros to keep close alignment between code and notation. This project aims to take a fresh start, using agentic models to consider alternative approaches and language combinations and better organise the codebase.

The folder notebooks contains initial experiments and notes which inform the design of this project.
