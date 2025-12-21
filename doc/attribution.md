# Attribution

The attribute function will calculate the significance of terms of an expression to the resulting prediction for probability of a given token.


For example, given a residual vector `x` at the position of the token "is" in the prompt "The capital of Ireland is", we can compute the attribution of each component to the logit for "Dublin":

We define a new symbol `▷` to represent attribution:
So if we have already expanded `x` into its components:
`T(x)` → `embed(x) + Δx^0 + Δx^1 + ... + Δx^{n-1}`
We can say:
`T(x)` ▷ `{ embed(x) : 20%, Δx^0 : 2%, Δx^1 : 50%+ ... + Δx^{n-1}: 1% }`
which reads as "the residual `T(x)` attributes 20% of the logit for 'Dublin' to the embedding of 'is', 2% to the contribution from layer 0, 50%+ to layer 1, and so on".

```python
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
x = T(" is")      # Pre-LN residual: T^n(embed(" world"))
expression = expand(x)   # Level 1 expansion - embed(x) + Δx^0 + Δx^1 + ... + Δx^{n-1}
attribute(expression, " Dublin")  # Compute attribution for logit of 'Dublin'
#returns an object which renders as:
# T(x) ▷ { embed(' is') : 20%, Δx^0 : 2%, Δx^1 : 50%+ ... + Δx^{n-1}: 1% }
```