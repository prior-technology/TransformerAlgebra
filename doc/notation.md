A key choice in this project is to treat network activations as abstract vectors and focus on relationships between them, instead of inspecting individual neuron activations in isolation.

- `context`\
   This encapsulates a specific language model and tokenizer, and any other model specific data required.   
 - `setup`\
   This expression defines symbols used in the expression depending on prompt text and any subsequent manipulation. It assumes context is available through a symbol ctx
 - `transform`\
   This should be a simple expression showing the operation of the transformer on a vector in the embedding space.
 - `interpretation`\
   This expression defines the output of the model (e.g. prediction of next token) using symbols from the previous stages.
