# DDM GP LOO Delta Procedure (Rank-1 Add/Remove)

## Scope and objective
This document defines the exact cluster-level LOO delta score used by DDM when moving one observation between mutually exclusive clusters.

We evaluate the change in the **total factorized LOO latent log density** for a cluster:

\[
\Delta_{\text{cluster}} = \mathcal L_{\text{new}} - \mathcal L_{\text{old}}.
\]

In the current implementation, the latent evaluation point is fixed to observations (equivalently, per-point residual argument is \(y_i-\mu_{-i}\)).

## Notation
For one cluster with \(n\) points:

\[
K_y = K + \sigma_n^2 I, \quad Q = K_y^{-1}, \quad \tilde y = y-m, \quad \alpha = Q\tilde y.
\]

LOO latent factors per point:

\[
\mu_i^{(f)} = y_i - \frac{\alpha_i}{Q_{ii}}, \qquad
v_i^{(f)} = \frac{1}{Q_{ii}} - \sigma_n^2.
\]

Per-point LOO latent log contribution:

\[
\ell_i = -\frac12\left[\frac{(y_i-\mu_i^{(f)})^2}{v_i^{(f)}} + \log v_i^{(f)} + \log(2\pi)\right],
\]
which is equivalent to
\[
\ell_i = \log \mathcal N\!\left(\frac{\alpha_i}{Q_{ii}};\,0,\,\frac{1}{Q_{ii}}-\sigma_n^2\right).
\]

Total score:

\[
\mathcal L = \sum_{i=1}^n \ell_i.
\]

## Add one point (rank-1 update)
Add \((x_+,y_+)\):

\[
k = K(X,x_+),\quad \kappa = k(x_+,x_+) + \sigma_n^2,
\]
\[
u = Qk,\quad s = \kappa - k^\top Qk,\quad
\delta = \tilde y_+ - k^\top\alpha,
\]
with \(\tilde y_+ = y_+ - m(x_+)\).

Block-inverse update:

\[
Q^+ = \begin{pmatrix}
Q + \frac{uu^\top}{s} & -\frac{u}{s} \\
-\frac{u^\top}{s} & \frac{1}{s}
\end{pmatrix},
\qquad
\alpha^+ = \begin{pmatrix}
\alpha + u\frac{\delta}{s} \\
\frac{\delta}{s}
\end{pmatrix}.
\]

For old points \(i=1,\dots,n\):

\[
Q^+_{ii} = Q_{ii}+\frac{u_i^2}{s},
\quad
\alpha_i^+ = \alpha_i + u_i\frac{\delta}{s}.
\]

So

\[
v_{i,+}^{(f)} = \frac{1}{Q^+_{ii}}-\sigma_n^2,
\quad
y_i-\mu_{i,+}^{(f)} = \frac{\alpha_i^+}{Q^+_{ii}}.
\]

For the new point:

\[
Q^+_{++}=\frac{1}{s},\quad \alpha_+^+=\frac{\delta}{s},
\quad
y_+ - \mu_{+,\mathrm{LOO}}^{(f)} = \delta,
\quad
v_{+,\mathrm{LOO}}^{(f)} = s-\sigma_n^2.
\]

Hence

\[
\mathcal L_{n+1} = \sum_{i=1}^n
\log \mathcal N\!\left(\frac{\alpha_i^+}{Q_{ii}^+};0,\frac{1}{Q_{ii}^+}-\sigma_n^2\right)
+
\log \mathcal N\!\left(\delta;0,s-\sigma_n^2\right).
\]

## Remove one point (rank-1 downdate)
Remove local index \(k\). Partition \(Q\) and \(\alpha\):

\[
Q = \begin{pmatrix}A & b \\ b^\top & d\end{pmatrix},
\qquad
\alpha = \begin{pmatrix}a\\ \alpha_k\end{pmatrix}
\]
(up to permutation placing removed index last).

Downdate identities:

\[
Q' = A - \frac{bb^\top}{d},
\qquad
\alpha' = a - b\frac{\alpha_k}{d}.
\]

Equivalent elementwise form in original indexing for remaining \(i\neq k\):

\[
Q'_{ii} = Q_{ii} - \frac{Q_{ik}^2}{Q_{kk}},
\qquad
\alpha'_i = \alpha_i - \frac{Q_{ik}}{Q_{kk}}\alpha_k.
\]

Then

\[
\mathcal L_{n-1} = \sum_{i\neq k}
\log \mathcal N\!\left(\frac{\alpha'_i}{Q'_{ii}};0,\frac{1}{Q'_{ii}}-\sigma_n^2\right).
\]

And the exact removal delta is

\[
\Delta_{\mathrm{remove}} = \mathcal L_{n-1} - \mathcal L_n.
\]

## Complexity
No explicit \(O(n^3)\) matrix inversion is used.

- Add: one solve \(u=Qk\) via LDLT (`solve`) \(O(n^2)\), then vector updates \(O(n)\).
- Remove: one solve for one inverse column \(q_{:k}=Qe_k\) \(O(n^2)\), then vector updates \(O(n)\).

## Code consistency audit (current repository)
Primary implementation: `include/model_cluster.h`, function `computeClusterLooDelta(...)`.

Checked items:
- Add update uses \(\alpha^+_{1:n} = \alpha + u\delta/s\): **YES**.
- Add update uses \(Q^+_{ii}=Q_{ii}+u_i^2/s\): **YES**.
- New-point add term uses \(\log \mathcal N(\delta;0,s-\sigma_n^2)\): **YES**.
- Remove downdate uses \(Q'_{ii}=Q_{ii}-Q_{ik}^2/Q_{kk}\): **YES**.
- Remove downdate uses \(\alpha'_i=\alpha_i-Q_{ik}\alpha_k/Q_{kk}\): **YES**.
- LOO latent variance uses \(1/Q_{ii}-\sigma_n^2\): **YES**.
- No `.inverse()` call in update/downdate path: **YES** (uses LDLT solves).

Operational note:
- The code contains defensive guards that return `0.0` if cluster state is not ready (`fit_` false or index/state mismatch). This prevents crashes but means no mathematical delta is applied in that invalid state. The formulas above assume valid conditioned/fitted state.

## Practical preconditions
For strict mathematical equivalence with this document, ensure before calling delta scoring:
- target cluster GP is conditioned/fitted for current memberships,
- local index table and cluster sizes are synchronized,
- \(s>0\) and all latent variances are positive (numerical floor may still be applied for robustness).
