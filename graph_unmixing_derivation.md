---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
---

**Graph Laplacian Estimation**

<!-- #region -->
**Graph Constrained Constrained Unmixing**


For simplicity, we denote the postive, sum-to-one constraint set as $\Delta = \{A | A \geq 0, 1_n^T A = 1_n  \}$. We express the A subproblem as follows:
$$
\min_{A} \frac{1}{2}\|MA-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(ALA^T) + i_{\Delta}(A)
$$
This has the following equivalent formulation:
$$
\begin{align*}
&\min_{U,V_1,V_2,V_2,V_4,V_5} \frac{1}{2}\|V_1-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + i_{\Delta}(V_2)\\


&\begin{align*}
 \text{subject to } V_1 &= MU\\
V_2 &= U\\
V_2 &= U\\
\end{align*}
\end{align*}
$$
which, in compact form, becomes
$$
\min_{U,V} g(V) \\\text{ subject to } GU + BV = 0
$$
where,
$$
g(V) = \frac{1}{2}\|V_1-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + i_{\Delta}(V_3)
$$

$$
V = \begin{bmatrix}
 V_1 &   &   \\ 
     &V_2&   \\ 
     &   &V_3\\ 
\end{bmatrix},
\quad
G = 
\begin{bmatrix}
M\\ 
I\\ 
I
\end{bmatrix},
\quad
B = 
\begin{bmatrix}
-I &  &  \\ 
   &-I&  \\ 
   &  &-I\\ 
\end{bmatrix}
$$

The augmented lagrangian $\mathcal{L}$ with parameter $\tau > 0$ of this problem is:
$$
\mathcal{L}(U,V,D) = g(V) + \frac{\tau}{2} \|GU + BV -D \|_F^2
$$
The corresponding ADMM updates are:
$$
\begin{align*}

U^{(k+1)} &= \argmin_U \frac{\tau}{2} \|GU + BV^{(k)} - D^{(k)} \|_F^2 \\
V^{(k+1)} &= \argmin_V g(V) + \frac{\tau}{2} \|GU^{(k+1)} + BV - D^{(k)} \|_F^2 \\
D^{(k+1)} &= D^{(k)} - GU^{(k+1)} - BV^{(k+1)}
\end{align*}

$$
By expanding V and D, we have
$$
\mathcal{L}(U,V_1,V_2,V_3,D_1,D_2,D_3) = \frac{1}{2}\|V_1-X\|_F^2 + \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + i_{\Delta}(V_3) + \frac{\tau}{2}\|MU-V_1-D_1\|_F^2 + \frac{\tau}{2}\|U-V_2-D_2\|_F^2 + \frac{\tau}{2}\|U-V_3-D_3\|_F^2
$$

The expanded but equivalent ADMM updates are:
$$
\begin{align*}
U^{(k+1)} &= \argmin_U \frac{\tau}{2}\|MU-V_1-D_1\|_F^2 + \frac{\tau}{2}\|U-V_2-D_2\|_F^2 + \frac{\tau}{2} \|U-V_3-D_3\|_F^2\\

V^{(k+1)}_1 &= \argmin_{V_1} \frac{1}{2}\|V_1-X\|_F^2 + \frac{\tau}{2} \|MU^{(k+1)} - V_1 - D^{(k)}_1 \|_F^2 \\
V^{(k+1)}_2 &= \argmin_{V_2} \frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + \frac{\tau}{2} \|U^{(k+1)} - V_2 - D^{(k)}_2 \|_F^2 \\
V^{(k+1)}_3 &= \argmin_{V_3} i_{\Delta}(V_3) + \frac{\tau}{2} \|U^{(k+1)} - V_3 - D^{(k)}_3 \|_F^2 \\

D^{(k+1)}_1 &= D^{(k)}_1 - MU^{(k+1)} - V^{(k+1)}_1 \\
D^{(k+1)}_2 &= D^{(k)}_2 - U^{(k+1)} - V^{(k+1)}_2 \\
D^{(k+1)}_3 &= D^{(k)}_3 - U^{(k+1)} - V^{(k+1)}_3
\end{align*}

$$

<!-- #endregion -->

# Deriving Updates

**U Update**

As the problem for the U update is convex and differentiable, we derive the KKT condition and solve for U accordingly:

$$

\begin{align*}

0 &= \frac{\partial}{\partial U}\left[\frac{\tau}{2}\|MU-V_1-D_1\|_F^2 + \frac{\tau}{2}\|U-V_2-D_2\|_F^2 + \frac{\tau}{2} + \|U-V_3-D_3\|_F^2\right]
\\
0 &= \tau \left(M^T(MU-V_1-D_1) + (U-V_2-D_2) + (U-V_3-D_3)\right)
\\
M^TMU + 2U &= M^T(V_1+D_1) + (V_2+D_2) + (V_3+D_3)
\\
U &= (M^T M + 2I)^{-1}(M^T(V_1+D_1) + (V_2+D_2) + (V_3+D_3)) 
\end{align*}
$$
The update for U is:
$$
U^{(k+1)} = \left(M^T M + 2I\right)^{-1}\left(M^T(V_1^{(k)}+D_1^{(k)}) + (V_2^{(k)}+D_2^{(k)}) + (V_3^{(k)}+D_3^{(k)})\right)
$$

**V1 Update**

For the $V_1$ update, since it is convex and differentiable, we derive the KKT condition and solve for $V_1$ directly:
$$
\begin{align*}
0 &= \frac{\partial}{\partial V_1} \left[ \frac{1}{2}\|V_1-X\|_F^2 + \frac{\tau}{2} \|MU - V_1 - D_1 \|_F^2 \right] \\
0 &= (V_1 - X) + \tau(V_1 - (MU - D_1)) \\
V_1 &= \frac{1}{1+\tau} \left(X + (MU - D_1)\right)
\end{align*}
$$
For the $V_1$ update, we have:
$$
V_1^{(k+1)} = \frac{1}{1+\tau} \left(X + (MU^{(k)} - D_1^{(k)})\right)
$$

**V2 Update**

For the $V_2$ update, it is both convex and differentiable, so we derive the KKT condition and solve for $V_2$ directly. We also note that, we can estimate $L = S \Sigma S^T$
$$
\begin{align*}

0 &= \frac{\partial}{\partial V_2} \left[\frac{\beta}{2} \mathrm{Tr}(V_2LV_2^T) + \frac{\tau}{2} \|U - V_2 - D_2 \|_F^2   \right] \\
0 &= \frac{\beta}{2}\left(V_2 \left(S \Sigma S^T \right)^T +  V_3 \left(S \Sigma S^T \right) \right) + \tau\left( V_2 - (U - D_2)\right) \\
0 &= \beta V_2 S \Sigma S^T + \tau V_2 - \tau (U - D_2) \\
V_2\left(S \Sigma S^T + \frac{\tau}{\beta} I\right) &= \frac{\tau}{\beta} (U - D_2) \\
V_2 &= \frac{\tau}{\beta} (U - D_2)\left(S \Sigma S^T + \frac{\tau}{\beta} I\right)^{-1} \\
V_2 &= \frac{\tau}{\beta} (U - D_2)S\left(\Sigma + \frac{\tau}{\beta}I\right)^{-1}S^T
\end{align*}

$$
The update for $V_2$ is simple as $(\Sigma + \frac{\tau}{\beta}I)$ is a diagonal matrix, so the inversion is given by taking the reciprocal of the entries. 
$$
V_2^{(k+1)} = \frac{\tau}{\beta} (U^{(k+1)} - D_2^{(k)})S\left(\Sigma + \frac{\tau}{\beta}I\right)^{-1}S^T
$$

**V3 Update**

For the $V_3$ update, as $\Delta$ is an affine set, the update simply involes a least squares projection on $\Delta$:
$$
V_3^{(k+1)} = \textbf{proj}_{\Delta}(U^{(k+1)} - D_3^{(k)})
$$

**A Subproblem Summary**

The final ADMM updates are:
$$
\begin{align*}
U^{(k+1)} &= \left(M^T M + 2I\right)^{-1}\left(M^T(V_1^{(k)}+D_1^{(k)}) + (V_2^{(k)}+D_2^{(k)}) + (V_3^{(k)}+D_3^{(k)})\right)\\

V^{(k+1)}_1 &= \frac{1}{1+\tau} \left(X + (MU^{(k)} - D_1^{(k)})\right) \\
V^{(k+1)}_2 &= \frac{\tau}{\beta} (U^{(k+1)} - D_2^{(k)})S\left(\Sigma + \frac{\tau}{\beta}I\right)^{-1}S^T \\
V^{(k+1)}_3 &= \textbf{proj}_{\Delta}(U^{(k+1)} - D_3^{(k)}) \\

D^{(k+1)}_1 &= D^{(k)}_1 - MU^{(k+1)} - V^{(k+1)}_1 \\
D^{(k+1)}_2 &= D^{(k)}_2 - U^{(k+1)} - V^{(k+1)}_2 \\
D^{(k+1)}_3 &= D^{(k)}_3 - U^{(k+1)} - V^{(k+1)}_3
\end{align*}

$$




TO DO:
complete updates for both A and M.
