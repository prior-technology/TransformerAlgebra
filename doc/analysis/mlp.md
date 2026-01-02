# Summary

We use $T^i(token)$ to represent the transformer with context acting on the last position embedding vector for token through all blocks up to block i, so it is the sum of \Delta x^0 to \Delta x^{i-1}. 

Then the input x to MLP block i is T^i(token) + attention output at block i.

We look at the "token-conditioned, activation-conditioned cone of sensitivity" of the MLP layer.

---

## 1. Formal definition (local, exact)

Fix:

* an input (x \in \mathbb{R}^d)
* a specific MLP block
* a specific output token (t)

Recall the logit contribution:

$$
\ell_t(x) = 

\sum_{i \in \mathcal A(x)}
(v_t)_i , (w_{1,i}\cdot x + b_{1,i})
$$

where:

* $(w_{1,i})$ = row $i$ of $W_1$
* $v_t = W_2^\top u_t$
* $\mathcal A(x) = { i : w_{1,i}\cdot x + b_{1,i} > 0 }$ = active neurons at input $x$

Now define the **effective sensitivity vectors**:

$$
g_i := (v_t)_i , w_{1,i}
\quad\text{for } i \in \mathcal A(x)
$$

Then locally (while the activation pattern is fixed):

$$
\nabla_x \ell_t(x)=\sum_{i \in \mathcal A(x)} g_i
$$

This gradient already *is* the cone collapsed to a single direction.
The cone is the **positive hull** that generates it.

---

## 2. The cone (precise object)

The **cone of sensitivity** is:


$$\mathcal C_t(x)
=
\left\{
\sum_{i \in \mathcal A(x)}
\alpha_i, g_i
;\middle|;
\alpha_i \ge 0
\right\}
\subset \mathbb R^d
$$

Properties:

* **Token-conditioned**: depends on $u_t$ via $v_t$
* **Activation-conditioned**: depends on which neurons are on
* **Polyhedral cone**: finitely generated
* **Low effective dimension**: usually tens of generators

This is not a subspace:

* no closure under negation
* no global linear structure

---

## 3. What vectors are “in the cone”?

A direction $\delta x$ is in the cone iff:

$$
\delta x = \sum_i \alpha_i g_i, \quad \alpha_i \ge 0
$$
Operational meaning:

* Moving $x \mapsto x + \epsilon,\delta x$ **monotonically increases** the logit (locally)
* Every generator corresponds to **one active neuron that matters for token $t$**

You can think of each $g_i$ as a *primitive explanation vector*.

---

## 4. Identifying the cone in practice (step-by-step)


### Step 1 — Fix locality

Choose a specific:

* sequence
* position
* forward pass
* activation pattern

Do **not** average across data.

---

### Step 2 — Identify active, relevant neurons

Compute:

* preactivations $a_i = w_{1,i}\cdot x + b_{1,i}$
* keep $i$ where:

  * $a_i > 0$
  * $|(v_t)_i|$ is non-negligible

This prunes aggressively.

---

### Step 3 — Form generators

For remaining neurons:

$$
g_i = (v_t)_i , w_{1,i}


You now have ~10–100 vectors in $\mathbb R^d$.

---

### Step 4 — Reduce redundancy

Compute SVD / PCA of the matrix with rows $g_i$:

* Rank is usually very small
* Directions cluster strongly

This gives you:

* a **basis for the cone’s span**
* principal sensitivity axes

---

## 5. Moving *within* the cone

### (A) Guaranteed monotonic moves

If you move along:

$$
\delta x = \nabla_x \ell_t(x)
$$

then for small $\epsilon$:

* logit strictly increases
* activation pattern stays fixed
* you remain inside the cone

This is the **central ray**.

---

### (B) Controlled exploration

To move without changing the active set:

You must satisfy, for all $i$:

$$
w_{1,i}\cdot (x + \delta x) + b_{1,i} > 0
$$

This defines **linear constraints**:

$$
w_{1,i}\cdot \delta x > -a_i
$$

So feasible moves lie in a **convex polytope** inside the cone.

In practice:

* small steps along any $g_i$ are safe
* mixtures are safer than single generators

---

### (C) What you cannot do

You cannot:

* flip signs arbitrarily
* move orthogonally to the cone and still affect the token
* cross ReLU boundaries without changing the cone itself

Once a neuron switches:

* generators appear/disappear
* cone changes discontinuously

---

## 6. Relation to “information depended on”

The cone answers precisely:

> **Which input perturbations does this token care about right now?**

Anything orthogonal to the cone:

* does not affect the logit to first order
* is “invisible” to this MLP for this token

Anything inside:

* is causally relevant
* corresponds to specific neuron features

---
