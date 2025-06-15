# I. Quick Introduction to Physics-Informed Neural Networks (PINNs)

Since PINNs are a really hot topic in science and in the deep learning community i will try in the following to give a quick introduction to the concept and to the main ideas behind it. I will not go into the details of the implementation or the mathematics behind it, for this you can refer to all the great papers and articles that are available freely on the subject [[1]](https://arxiv.org/abs/2005.04593) [[2]](https://arxiv.org/abs/2005.04593)
[[3]](https://arxiv.org/abs/2007.06007)

Physics-Informed Neural Networks (PINNs) are a class of deep learning models that are used to solve a partial differential equation or to regularized data given a physical behavior. This is a really smart and lightweight way of dealing with these problems and unearths a lot of possibilities for every physical fields.

### I.1  NN representation and Constraints
We will assume in the following that the concept of NN and MLP are known by the reader. For the sake of clarity we will recall that a traditional MLP NN applied to an input $\mathbf{x}$  can be written in the following form :

$$
	\Psi^{\theta_{\vec{n}}}(\mathbf{x})
	=
	W^{(L)}\sigma\Bigl(
	\mathbf{W}^{(L_1)} \,\sigma(\dots \sigma(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})) + \mathbf{b}^{(L)}
	\Bigr)
$$

Where $\sigma$ is the activation function of the $l-1$ layers applied element wise to the input matrix and $W^{(l)}, \, b^{l}$ are respectfuly the weight and the bias matrix of the layer $l$.

That might look barbarous at first glance but you can see this as a parametrized function that we will use to explore our solution space with well define and easy to compute derivatives. For exmample let's assume that we want to simply fit a function
$$f : \mathbb{R} \to \mathbb{R}$$

at  given poins $x_1 \dots x_N$. Then you want the neural network to verify :

$$ \Psi^{\theta_{\vec{n}}}(x_i) = f(x_i), \, \forall i \in [1,N]$$

This kind of representation where each element of the input space can be evaluated by $\Psi^{\theta_{\vec{n}}}$ is called an implicit representation, and it allows us to have a *general approximator* of the function $f$.

The reader will understand that this kind of tools are really useful when we don't know the function $f$ at all the points, but we know some constraints on this function for example, as previously, that it takes some values at given points. Then $\Psi^{\theta_{\vec{n}}}$ will be a perfect surrogate to the function $f$. This kind of method are really close to polynomial or Fourier interpolation but we will see further that it allows us to have more freedom in our computations.

However, one does not have to forget that this Neural Network surrogate leads to a first approximation of the function $f$ with a given error. This is generally called the *approximation error* and it is a function of the number of layers and the number of neurons in each layer. This is a really important point to keep in mind when using PINNs, since the approximation error will be added to the numerical error that we will introduce with the optimization algorithm.

We can obviously impose more than just point-wise constraints, and for example add a differential equations constraints, this is the main background of PINNs. For example let's illustrate that with a simple wave equation: :
$$
	\frac{\partial^2 u}{\partial t^2} + c^2 \frac{\partial^2 u}{\partial x^2} = 0, \quad \forall (t,x) \in [0,T] \times [0,L]
$$
Where $c$ is a constant. Given the finite interval, if we want to have a unique solution we need to add some initial and boundary conditions. This can be also viewed as point-wise constraints for the function $u$.
$$
	u(0,x) = f(x), \quad \forall x \in [0,L]
$$
$$
	u(t,0) = b(t), \quad \forall t \in [0,T]
$$

Where $f$ and $b$ are two functions. Given all of this we can now define the total constraints as the following :

$$
	\mathcal{C} = \Bigl\{ u(0,x) - f(x), \, u(t,0) - b(t), \, \frac{\partial^2 u}{\partial t^2} + c^2 \frac{\partial^2 u}{\partial x^2} , (x,t) \in [0,L] \times [0,T]\Bigr\}
$$

This is a set of constraints that we want to minimize. This kind of multi-objective optimization problem is actually a really lively field of reasearch by the time of this post, but for the sake of simplicity we will only consider the following linear scalarization of the problem:

$$\begin{align*}
	\mathcal{L}(\theta_{\vec{n}}) = \int_{0}^{L} \Bigl( u(0,x) - f(x) \Bigr)^2 dx + \int_{0}^{T} \Bigl( u(t,0) - b(t) \Bigr)^2 dt + \\
	+\int_{0}^{T} \int_{0}^{L} \Bigl( \frac{\partial^2 u(t,x)}{\partial t^2} + c^2 \frac{\partial^2 u(t,x)}{\partial x^2} \Bigr)^2 dx dt
\end{align*}
$$

Since we defined a loss it's then really easy to fall back to the traditional NN framework.  The next step is then to apply a gradient descent algorithm to our neural network parameters $\theta_{\vec{n}}$ to minimize the loss function $\mathcal{L}(\theta_{\vec{n}})$.

$$
	\theta_{\vec{n}}^{(k+1)} = \theta_{\vec{n}}^{(k)} - \eta \nabla_{\theta_{\vec{n}}} \mathcal{L}(\theta_{\vec{n}}^{(k)})
$$

Where $\eta$ is the learning rate. This is thy every basic of optimization algorithm, and we will not go into the details of the optimization algorithm used in the following. Obviously, the loss function $\mathcal{L}$ can not be evaluated in its integral form sinve we oftenly do not have access to the constraints value at every  point, this is why we evaluate the loss function using a simple Monte Carlo approximation.

For example given $N_i$ points of initial condition $\{(0, x_j)\}_{j=1}^{N_i}$, $2N_b$ points of boundary $\{(t_j, 0)\}_{j=1}^N$,  $\{(t_j, L)\}_{j=1}^N$ and $N_{pde}$ collocation points for the PDE $\{(t_k,x_l)\}_{k,l=1}^N$ we can approximate the loss function as follows:

$$
	\mathcal{L}(\theta_{\vec{n}}) \approx \frac{1}{N_d} \sum_{i=1}^{N_d} \Bigl( u(0,x_i) - f(x_i) \Bigr)^2 + \frac{1}{N_b} \sum_{j=1}^{N_b} \Bigl( u(t_j,0) - b(t_j) \Bigr)^2 + \\
	+\frac{1}{N_{pde}} \sum_{k,l=1}^{N_{pde}} \Bigl( \frac{\partial^2 u(t_k,x_l)}{\partial t^2} + c^2 \frac{\partial^2 u(t_k,x_l)}{\partial x^2} \Bigr)^2
$$

This simplification leads to a unbiased approximation of the loss function, with a variance that is proportional to the number of points used in the approximation, the error we made on the integral value is $\mathcal{O}(1/\sqrt{N})$ where $N$ is the number of points used in the approximation. So in addition to the previous *approximation error* we have a first *estimation error* resulting from the Monte Carlo approximation of the loss function, this is also really important to keep in mind when using PINNs.


{{% callout note %}}
      This post is still under writing and will be updated in the following days. I will try to add some more details on the implementation and the mathematics behind PINNs.
{{% /callout %}}
