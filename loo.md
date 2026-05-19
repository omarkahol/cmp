---
noteId: "e9f398b0303d11f188639be8e9c8b113"
tags: []

---

# What I extracted from ChatGPT

Yes. Let me write the **exact closed-form update** for the **factorized total LOO log density of the latent variables** after adding one new observation.

I will keep the mean function nonzero and fixed.

---

## 1. Setup before adding the new point

Current dataset:
$$
\mathcal D_n={(x_i,y_i)}_{i=1}^n.
$$

Model:
$$
f \sim \mathcal N(m,K), \qquad y\mid f \sim \mathcal N(f,\sigma_n^2 I).
$$

Define the current GP matrix and inverse
$$
K_y = K + \sigma_n^2 I,\qquad Q = K_y^{-1},$$
"centered" observations and regression coefficients
$$
\tilde y = y-m,\qquad \alpha = Q\tilde y.
$$

For the old (n)-point dataset, the LOO latent factors are
$$
f_i\mid \mathcal D_{n,-i}\sim \mathcal N(\mu_i^{(f)},v_i^{(f)}),
$$
with the current LOO predictive mean and variance (at the point left)
$$
\mu_i^{(f)} = y_i-\frac{\alpha_i}{Q_{ii}},\qquad
v_i^{(f)}=\frac{1}{Q_{ii}}-\sigma_n^2.
$$

So the old total factorized LOO latent log density, evaluated at some latent vector
$
f^{\rm eval}_{1:n},
$
is
$$
\mathcal L_n=
-\frac12\sum_{i=1}^n
\left[
\frac{(f_i^{\rm eval}-\mu_i^{(f)})^2}{v_i^{(f)}}
+\log v_i^{(f)}
+\log(2\pi)
\right].
$$

---

## 2. Add one new observation

Add a new observation $(x_+,y_+)$.
Define the vector and scalar
$$
k = K(X,x_+) \in \mathbb R^n,
\qquad
\kappa = k(x_+,x_+) + \sigma_n^2,
$$
as well as the new point's prior mean and "centered" observation 
$$
m_+ = m(x_+),\qquad
\tilde y_+ = y_+ - m_+.
$$
Also define the standard online-GP quantities
$$
u = Qk,
\qquad
s = \kappa - k^\top Qk,
\qquad
\delta = \tilde y_+ - k^\top \alpha.
$$

Then the updated inverse is
$$
Q^+ =
\begin{pmatrix}
Q + \dfrac{uu^\top}{s} & -\dfrac{u}{s}\cr \cr
-\dfrac{u^\top}{s} & \dfrac{1}{s}
\end{pmatrix},
$$
and the updated coefficient vector is
$$
\alpha^+ =
\begin{pmatrix}
\alpha + u\dfrac{\delta}{s}\cr
\dfrac{\delta}{s}
\end{pmatrix}.
$$

---

## 3. Updated LOO latent factors

### For the old points $(i=1,\dots,n)$

The updated diagonal entries are
$$
Q^+_{ii} = Q_{ii} + \frac{u_i^2}{s}.
$$

Hence the updated LOO latent variance is
$$
v_{i,+}^{(f)} =\frac{1}{Q_{ii}+\dfrac{u_i^2}{s}}-\sigma_n^2
=
\frac{s}{sQ_{ii}+u_i^2}-\sigma_n^2.
$$

The updated LOO latent mean is
$$
\mu_{i,+}^{(f)}
= y_i-\frac{\alpha_i + u_i\delta/s}{Q_{ii}+u_i^2/s}
=
y_i-\frac{s\alpha_i+u_i\delta}{sQ_{ii}+u_i^2}.
$$

So for each old point,
$$
f_i\mid \mathcal D_{n+1,-i}
\sim
\mathcal N\left(
y_i-\frac{s\alpha_i+u_i\delta}{sQ_{ii}+u_i^2},
\frac{s}{sQ_{ii}+u_i^2}-\sigma_n^2
\right).
$$

### For the new point (i=n+1)

Since
$$
Q^+_{n+1,n+1}=\frac{1}{s},\qquad
\alpha^+_{n+1}=\frac{\delta}{s},
$$
we get
$$
\mu_{+,{\rm LOO}}^{(f)}
= y_+ - \frac{\alpha^+_{n+1}}{Q^+_{n+1,n+1}}
= y_+ - \delta
m_+ + k^\top \alpha,
$$
and
$$
v_{+,{\rm LOO}}^{(f)}
= \frac{1}{Q^+_{n+1,n+1}}-\sigma_n^2=
s-\sigma_n^2.
$$

This is exactly the old latent predictive distribution at $(x_+)$:
$$
f_+\mid \mathcal D_n
\sim
\mathcal N\bigl(m_+ + k^\top\alpha,; s-\sigma_n^2\bigr).
$$

---

## 4. Closed-form updated total LOO latent log density

Now evaluate the factorized LOO latent log density at an extended latent vector
$$
f^{\rm eval,+}=(f_1^{\rm eval},\dots,f_n^{\rm eval},f_+^{\rm eval}).
$$

Then the exact updated total is

$$
\begin{align}
\mathcal L_{n+1}
=&-\frac12
\sum_{i=1}^n
\left[
\frac{\left(f_i^{\rm eval}-y_i+\dfrac{s\alpha_i+u_i\delta}{sQ_{ii}+u_i^2}\right)^2}
{\dfrac{s}{sQ_{ii}+u_i^2}-\sigma_n^2}\right.  \cr
&+ \left.
\log\:
\left(\dfrac{s}{sQ_{ii}+u_i^2}-\sigma_n^2 \right)
+
\log(2\pi)
\right]
\cr
&-\frac12
\left[
\frac{\left(f_+^{\rm eval}-m_+-k^\top\alpha\right)^2}{s-\sigma_n^2,}
+
\log(s-\sigma_n^2)
+
\log(2\pi)
\right].
\end{align}
$$

That is the closed-form expression.

---

## 5. Difference form relative to the old total

If you want it as an update from the old total $\mathcal L_n$, then

$$
\mathcal L_{n+1}=\mathcal L_n+\Delta_{\rm old}+\Delta_{\rm new},
$$
where
$$
\Delta_{\rm old} = 
-\frac12\sum_{i=1}^n
\left[
\frac{(f_i^{\rm eval}-\mu_{i,+}^{(f)})^2}{v_{i,+}^{(f)}}
-
\frac{(f_i^{\rm eval}-\mu_i^{(f)})^2}{v_i^{(f)}}
+
\log\frac{v_{i,+}^{(f)}}{v_i^{(f)}}
\right],
$$
and
$$
\Delta_{\rm new}
=
-\frac12
\left[
\frac{(f_+^{\rm eval}-m_+-k^\top\alpha)^2}{s-\sigma_n^2}
+\log(s-\sigma_n^2)
+\log(2\pi)
\right].
$$

This form is useful if $\mathcal L_n$ has been computed and is stored.

---

## 6. Computational cost

Given the old state $(Q,\alpha)$ or equivalently a factorization plus solves:

* compute $u=Qk$: $O(n^2)$
* compute $s$: $O(n)$
* compute $\delta$: $O(n)$
* update all $n$ old LOO terms: $O(n)$

So once the new-point quantities are available, updating the **whole factorized total LOO latent log density** is $O(n)$ on top of the $O(n^2)$ GP update.

---

