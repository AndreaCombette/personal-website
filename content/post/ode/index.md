---
title:  Numerical methods for ordinary differential equations
subtitle:  most basic numerical methods to solve Cauchy problem
summary:  most basic numerical methods to solve Cauchy problem
authors:
  - Guillaume Legendre 
  - admin

tags: []
categories: []
projects: []
date: '2023-07-05T00:00:00Z'
lastMod: '2023-07-05T00:00:00Z'
image:
  caption: ''
  focal_point: ''
---

# Tutorial. Numerical methods for ordinary differential equations

A general technique for solving problems with partial differential equations (PDE) that involve a time variable $t$ and one or more space variables is the so-called [method of lines](https://en.wikipedia.org/wiki/Method_of_lines). It proceeds by first discretising the partial derivatives with respect to the space variables and leaving the time variable continuous, leadind to an approximating system of ordinary differential equations (ODE) in the variable $t$. The Cauchy problem for the resulting ordinary differential equations is then solved using a standard method for the task.

The convenience and versatility of readily available ordinary differential equation solvers permits complicated partial differential equations to be solved with comparative ease in this manner, especially when there is only one space variable. Note that it might well be more efficient to solve a given partial differential equation by a method particularly suited to the equation, but the easy solution of rather general problems with acceptable accuracy makes the approach of the method of lines attractive.

In the present notebook, we are interested in reviewing some of the most basic numerical methods to solve Cauchy problems for an ordinary differential equation of the form:
{{< math >}}
$$
\begin{cases}
u'(t)=f(t,u(t)),\ t>0,\\
u(0) = u_0,
\end{cases}
$$
{{< /math >}}
where $u$ is a function typically taking its values in $\mathbb{R}^d$ (in the method of lines, the integer $d$ is the dimension of the semi-discretised partial differential equation problem).

For this notebook, we will need the <tt>numpy</tt> and <tt>matplotlib</tt> packages.


```python
import numpy as np

# To draw matplotlib plots within this notebook.
import matplotlib.pyplot as plt
from python_code.nord_cmap import * # import mycmap
```

## Exercise 1. The SIR model.
Consider the system of the Kermack-McKendrick [SIR model](http://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model), describing the evolution of a population affected by an infectious disease
{{< math >}}
$$
\left\{
\begin{align*}
S'(t)&=-r\,S(t)I(t),\\
I'(t)&=r\,S(t)I(t)-a\,I(t),\\
R'(t)&=a\,I(t),
\end{align*}
\right.\ t>0,
$$
{{< /math >}}
associated with the initial data
$$
S(0)=S_0,\ I(0)=I_0,\ R(0)=R_0.
$$
The functions $S$, $I$, and $R$ represent the numbers of people respectively susceptible to the disease, infected by the disease, and immune after being infected over time, the real number $r$ is the infection rate, and the real number $a$ is the rate of recovery. We will use the following values for the problem parameters:

$$
S_0=762,\ I_0=1,\ R_0=0,\ r=0.00218,\ a=0.44036,\ T=14,
$$

obtained by calibrating the model using data from an influenza epidemic in a boys' school published in the British medical journal *The Lancet* on March 4, 1978.

**Question.** Compute an approximate solution of the system on the time interval $[0,T]$ using the [explicit (or forward) Euler method](https://en.wikipedia.org/wiki/Euler_method) and $N=10^5$ discretisation steps. To do this, write an independent function `explicit_euler` implementing the method (it will be used in other exercises), whose inputs are the function $f$ in the right-hand side of the ODE, the initial and final values of the time variable, the initial datum of the problem, the number of steps used for the uniform discretisation grid, and whose outputs are two arrays containing respectively the values of the times at which the approximation of the solution is computed and the corresponding values of this approximation. Represent the time evolution of the values of the approximations obtained for the numbers $S(t)$, $I(t)$ and $R(t)$.


```python
def explicit_euler(f, final_time, init_time, init_data, N_steps) :
    
    # allocation of memory
    
    integrated_data = np.zeros(shape = (N_steps, len(init_data)))
    integrated_data[0] = init_data
    h = (final_time - init_time) / N_steps
    
    for i in range(1, N_steps) : 
        
        integrated_data[i] = integrated_data[i-1] + h * f(h*i,integrated_data[i-1])
    
    return integrated_data
        

```


```python
r,a=0.00218,0.44036
def f(t,u): return np.array([-r*u[0]*u[1],r*u[0]*u[1]-a*u[1],a*u[1]])
```


```python


u0=np.array([762.,1.,0])

data = explicit_euler(f, 14, 0, u0, 10**5)
parameter_name = ['S', 'I', 'R']
fig, ax = plt.subplots()
color = color_list(3)

time = np.linspace(0,14, 10**5)

for k in range(len(u0)) : 
    
    ax.plot(time, data[:,k], color = color[k], label = parameter_name[k])
    ax.set_xlabel('time')
    ax.set_ylabel('Population')
    
    ax.set_title('Results of EE integration')
    ax.legend()
    
```


    
![png](./tutorial_ode_5_0.png)
    


**Question.** Let $N$ denote the number of discretisation steps used. Check theoretically and numerically that the obtained approximation verifies, for all $0\leq n\leq N$,
$$
S_n+I_n+R_n=S_0+I_0+R_0.
$$

**Answer.**


```python
Sum = np.sum(data, axis = 1)
checking = Sum - Sum[0]
fig, ax = plt.subplots()
ax.plot(checking, color = color[1], lw = .2)

ax.set_xlabel('number iterations')
ax.set_ylabel('$S_n + I_n + R_n$')
```




    Text(0, 0.5, '$S_n + I_n + R_n$')




    
![png](./tutorial_ode_8_1.png)
    


**Question.** Taking as reference solution the one obtained with $N=10^5$ discretisation steps, represent (using a logarithmic scale) the errors at time $T$ found using successively $N=10,10^2,\dots,10^4$ discretisation steps and determine the effective order of convergence of the method using the ``polyfit`` function from <tt>numpy</tt>.


```python
N = np.arange(10,10**4, 50)
all_predicted_data =[]
errors = []

for Num_steps in N :
    
    all_predicted_data.append(explicit_euler(f, 14, 0, u0, Num_steps))
    errors.append(np.linalg.norm(data[-1] - all_predicted_data[-1][-1],2))
    
tau = 1/N
fig, ax = plt.subplots()
ax.plot(np.log(tau), np.log(errors), 'o', color = color[1], markersize = 3)
fit = np.polyfit(np.log(tau), np.log(errors), deg  = 1)
poly = np.poly1d(fit)
num = np.linspace(10,10**3,10000)
ax.plot(np.log(tau), poly(np.log(tau)), '--', color = color[2], label = f'fit  : y = {fit[0]} x + {fit[1]}')
ax.set_xlabel('power of discretization')
ax.set_ylabel('errors')
ax.legend()
ax.set_title('Order of Convergence')


```




    Text(0.5, 1.0, 'Order of Convergence')




    
![png](./tutorial_ode_10_1.png)
    


**It's a linear convergence, first coefficient of the fit is 1**

**Question.** Write, on the model of the `explicit_euler` function, a function named `rk4` for the ["classic" Runge--Kutta method of order four](https://en.wikipedia.org/wiki/Runge–Kutta_methods#The_Runge–Kutta_method). The Butcher tableau of the method is
{{< math >}}
$$
\begin{array}{c|cccc}
0\\
\frac{1}{2}&\frac{1}{2}\\
\frac{1}{2}&0&\frac{1}{2}\\
1&0&0&1\\
\hline
&\frac{1}{6}&\frac{1}{3}&\frac{1}{3}&\frac{1}{6}
\end{array}.
$$
{{< /math >}}
Determine the effective order of this method by proceeding as in the previous question.


```python
def rk4(f, final_time, init_time, init_data, N_steps) : 
    
    integrated_data = np.zeros(shape = (N_steps, len(init_data)))
    integrated_data[0] = init_data
    h = (final_time - init_time) / N_steps
    
    for i in range(1, N_steps) : 
        
        time = i*h
        
        k1 = f(time, integrated_data[i-1])
        k2 = f(time + h/2, integrated_data[i-1] + h * k1/2)
        k3 = f(time + h/2, integrated_data[i-1] + h * k2/2)
        k4 = f(time + h, integrated_data[i-1] + h * k3)
        
        coeffs = np.array([1/6 * k1, 1/3 * k2, 1/3 * k3, 1/6 * k4])        
        Butcheur_sum = np.sum(coeffs, axis = 0)
        integrated_data[i] = integrated_data[i-1] + h * Butcheur_sum
        
    return integrated_data
```


```python
data = rk4(f, 14, 0, u0, 10**5)
parameter_name = ['S', 'I', 'R']
fig, ax = plt.subplots()
color = color_list(3)
u0=np.array([762.,1.,0])
time = np.linspace(0,14, 10**5)

for k in range(len(u0)) : 
    
    ax.plot(time, data[:,k], color = color[k], label = parameter_name[k])
    ax.set_xlabel('time')
    ax.set_ylabel('Population')
    
    ax.set_title('Results of RK4 integration')
    ax.legend()
```


    
![png](./tutorial_ode_14_0.png)
    



```python
N = np.arange(10,10**4, 10)
all_predicted_data =[]
errors = []

for Num_steps in N :
    
    all_predicted_data.append(rk4(f, 14, 0, u0, Num_steps))
    errors.append(np.linalg.norm(data[-1] - all_predicted_data[-1][-1],2))
    
tau = 1/N
fig, ax = plt.subplots()
ax.plot(np.log(tau), np.log(errors), 'o', color = color[1], markersize = 3)
fit = np.polyfit(np.log(tau), np.log(errors), deg  = 1)
poly = np.poly1d(fit)

ax.plot(np.log(tau), poly(np.log(tau)), '--', color = color[2], label = f'fit  : y = {fit[0] : .2f} x + {fit[1]: .2f}')
ax.set_xlabel('power of discretization')
ax.set_ylabel('errors')
ax.legend()
ax.set_title('Order of Convergence')

```




    Text(0.5, 1.0, 'Order of Convergence')




    
![png](./tutorial_ode_15_1.png)
    


## Exercise 2. The Dahlquist test equation.

The goal of this exercise is to use several one-step methods to numerically solve a problem based on the Dahlquist test equation, that is the following autonomous differential equation:
$$
u'(t)=\alpha u(t),\ t>0,
$$
where $u$ takes its values in $\mathbb{C}$ and $\alpha$ belongs to $\mathbb{C}$, completed by the initial condition
$$
u(0) = 1,
$$
up to a given final time $T$. The unique solution of this Cauchy problem possesses a closed-form expression, given by
$$
u(t) = e^{\alpha t}.
$$
When $\text{Re}(\alpha)<0$, this problem is at the basis of the notion of [A-stability](https://en.wikipedia.org/wiki/Stiff_equation#A-stability) for the behaviour of methods applied to the numerical solution of [stiff equations](https://en.wikipedia.org/wiki/Stiff_equation).

In what follows, we denote by $N$ the number of steps in the uniform grid discretising the interval $[0,T]$ and by $\Delta t=\frac{T}{N}$ the length of a step. We will denote by $u^{(n)}_{\text{*}}$ the numerical approximation of the exact value $u(n \Delta t)$,  where the character * refers to the selected method.

The chosen values of the parameters are $T=10$ and $N=100$.


```python
T=10.
N=100
```

**Question.** Using each of the following methods, numerically solve the Cauchy problem and plot both the numerical solution and the exact solution for several (both positive and negative) real values of $\alpha$. Are some numerical instabilities observed?

**a. The explicit Euler method.** For this method, use the function written in the previous exercise.


```python
alphas = np.linspace(-4,4,5)
u0=np.array([1.])
fig, ax = plt.subplots()
color = color_list(5)
time = np.linspace(0,T, N)


for k, alpha  in enumerate(alphas) : 
    
    def f(t,u): return alpha*u
    data = explicit_euler(f, T, 0, u0, N)

    ax.loglog(time, data[:,0], 'o', color = color[k], label = rf'$ u_* : \alpha = $ {alpha : .2f}', markersize = 1)
    ax.loglog(time, np.exp(alpha*time), '-', color = color[k], label = rf'$u : \alpha = $ {alpha: .2f}')
    ax.set_xlabel('time')
    ax.set_ylabel('$u_{EE}(t)$')
    
    ax.set_title('Results of EE integration')
    ax.legend()
```


    
![png](./tutorial_ode_20_0.png)
    


**For extremal values of alpha, aberrations can be observed.**

**b. The implicit Euler method.** Recall that the [implicit (or backward) Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method) is another first-order one-step method, defined by the recurrence relation
{{< math >}}
$$
u^{(n+1)}_{\text{IE}}=u^{(n)}_{\text{IE}}+h\,f((n+1)h,u^{(n+1)}_{\text{IE}}),
$$
{{< /math >}}
in which $h$ is the fixed steplength. To determine $u^{(n+1)}_{\text{IE}}$ at each step, the (possibly nonlinear) equation above has to be solved numerically, using for instance the `optimize.fsolve` function from <tt>scipy</tt>.


```python
from scipy.optimize import fsolve

def implicit_EE(f, final_time, init_time, init_data, N_steps) : 
    
    integrated_data = np.zeros(shape = (N_steps, len(init_data)))
    integrated_data[0] = init_data
    h = (final_time - init_time) / N_steps
    
    for i in range(1,N_steps) : 
        
        def equation(x) : return integrated_data[i-1] - x + h * f (h*i, x)
        
        integrated_data[i] = fsolve(equation, init_data)
        
    return integrated_data
        
```


```python
alphas = np.linspace(-4,4,5)
u0=np.array([1.])
fig, ax = plt.subplots()
color = color_list(5)
time = np.linspace(0,T, N)


for k, alpha  in enumerate(alphas) : 
    
    def f(t,u): return alpha*u
    data = implicit_EE(f, T, 0, u0, N)

    ax.loglog(time, data[:,0], 'o', color = color[k], label = rf'$ u_* : \alpha = $ {alpha : .2f}', markersize = 1)
    ax.loglog(time, np.exp(alpha*time), '-', color = color[k], label = rf'$u : \alpha = $ {alpha: .2f}')
    ax.set_xlabel('time')
    ax.set_ylabel('$u_{EE}(t)$')
    
    ax.set_title('Results of IE integration')
    ax.legend()
```

    /home/andrea/anaconda3/lib/python3.11/site-packages/scipy/optimize/_minpack_py.py:178: RuntimeWarning: The iteration is not making good progress, as measured by the 
      improvement from the last ten iterations.
      warnings.warn(msg, RuntimeWarning)



    
![png](./tutorial_ode_24_1.png)
    


**Answer.**

**c. The trapezoidal rule.** The [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)) is an implicit one-step method, which can be considered as both a [Runge–Kutta method](https://en.wikipedia.org/wiki/Runge–Kutta_methods) and a [linear multistep method](https://en.wikipedia.org/wiki/Linear_multistep_method) of Adams-Moulton type, defined by the following recursion relation
{{< math >}}
$$
u^{(n+1)}_{\text{TR}}=u^{(n)}_{\text{TR}}+\frac{h}{2}\,\left(f(nh,u^{(n)}_{\text{TR}})+f((n+1)h,u^{(n+1)}_{\text{TR}})\right),
$$
{{< /math >}}
in which $h$ is the fixed spacing.

**Preliminary question.** Show that the trapezoidal rule method is a second-order method.

It follows from the error analysis of the trapezoidal rule for quadrature that the local truncation error $\tau _{n}$ of the trapezoidal rule for solving differential equations can be bounded as:

$|\tau _{n}|\leq {\tfrac {1}{12}}h^{3}\max _{t}|y'''(t)|.$

**Answer.**


```python
def trapezoidal_rule(f, final_time, init_time, init_data, N_steps) : 
    
    integrated_data = np.zeros(shape = (N_steps, len(init_data)))
    integrated_data[0] = init_data
    h = (final_time - init_time) / N_steps
    
    for i in range(1,N_steps) : 
        
        def equation(x) : return integrated_data[i-1] + h/2 * f (h*(i-1), integrated_data[i-1]) - x + h/2 * f (h*i, x)
        
        integrated_data[i] = fsolve(equation, init_data)
        
    return integrated_data
```


```python
alphas = np.linspace(-4,4,5)
u0=np.array([1.])
fig, ax = plt.subplots()
color = color_list(5)
time = np.linspace(0,T, N)


for k, alpha  in enumerate(alphas) : 
    
    def f(t,u): return alpha*u
    data = trapezoidal_rule(f, T, 0, u0, N)

    ax.loglog(time, data[:,0], 'o', color = color[k], label = rf'$ u_* : \alpha = $ {alpha : .2f}', markersize = 1)
    ax.loglog(time, np.exp(alpha*time), '-', color = color[k], label = rf'$u : \alpha = $ {alpha: .2f}')
    ax.set_xlabel('time')
    ax.set_ylabel('$u_{EE}(t)$')
    
    ax.set_title('Results of Trapezoidal rule integration')
    ax.legend()
```


    
![png](./tutorial_ode_30_0.png)
    


**Answer.**

## Exercise 3. A second-order differential equation: the harmonic oscillator.

We consider the following Cauchy problem:
$$\begin{cases}
u''(t)+\omega^2u(t)=0,\ t>0,\\
u(0)=1,\\
u'(0)=0,
\end{cases}
$$
to be solved up to time $T$, where $u$ takes its values in $\mathbb{R}$, $\omega$ being a given real number.

**Question.** Introducing the function $v=\frac{u'}{\omega}$, show that this system is equivalent to a complex exponential system such as defined in the previous exercise. Express $\alpha$ in terms of $\omega$.

Introducing the complex explonential $u = e^{iwt}$, we reach the following equation : 

$$\begin{cases}
v'(t) = i \omega v(t),\ t>0,\\
u(0)=1,\\
u'(0)=0,
\end{cases}
$$


**Question.** Solve numerically the resulting system for $\omega = 1$ using the explicit Euler method, the implicit Euler method, and the trapezoidal rule method.


```python
omega=1.
alpha=omega*np.array([[0.,-1.],[1.,0.]])

def f(t,y): return alpha.dot(y)

T=20.
N=500

u0=np.array([1.,0.])


```

**Question.** Plot the respective [phase portraits](https://en.wikipedia.org/wiki/Phase_portrait) of the numerical approximations (that is, plot the approximations in the $(u,v)$-plane).


```python
data_1 = explicit_euler(f, 10, 0, u0, 10**2)
data_2 = implicit_EE(f, 10, 0, u0, 10**2)
data_3 = trapezoidal_rule(f, 10, 0, u0, 10**2)

fig, ax = plt.subplots()
color = color_list(3)

time = np.linspace(0,10, 10**2) 
    
ax.plot(data_1[:,0], data_1[:,1], color = color[0], label = 'EE', lw = 2)
ax.plot(data_2[:,0], data_2[:,1], color = color[1], label = 'IE', lw = 2)
ax.plot(data_3[:,0], data_3[:,1], color = color[2], label = 'Trap', lw = 2)

ax.set_xlabel('u')
ax.set_ylabel('v')

ax.set_title('Approximated solutions ')
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fc9fbbd9c90>




    
![png](./tutorial_ode_37_1.png)
    


**Question.** If one wants to enforce energy conservation in this system, which numerical method should be preferably used? Explain theoretically what is observed with the selected numerical method.

The only method enforcing the energy conservation is the trapezoidal method. We observe in the phase portrait the evolution of the pendulum speed in function of its position and without external constraints it the phase portait must me a closed contour (no revolution)

## Exercise 4. Stiff equation and absolute stability.
We want to approximate numerically the solution to the Cauchy problem
$$
\begin{cases}
u'(t)=-50\,\left(u(t)-\sin(t)\right)+\cos(t),& t>0,\\
u(0)=1,
\end{cases}
$$
on the interval $[0,2]$. 

**Question :** Use the explicit Euler method to solve the problem with a discretisation step of constant length, successively chosen

- strictly greater than 0.04, 
- between 0.02 and 0.04,
- strictly less than 0.02.

What is observed and what explanation can be given?


```python

def f(t,u): return -50*(u-np.sin(t))+np.cos(t)

u0=np.array([1.])

time_1 = np.linspace(0,2, 40) 
data_1 = explicit_euler(f, 2, 0, u0, 40)

time_2 = np.linspace(0,2, 80) 
data_2 = explicit_euler(f, 2, 0, u0, 80)

time_3 = np.linspace(0,2, 120) 
data_3 = explicit_euler(f, 2, 0, u0, 120)

fig, ax = plt.subplots()
color = color_list(3)

    
ax.loglog(time_1, data_1[:,0], color = color[0], label = f'$h =  0.05$', lw = 2)
ax.loglog(time_2, data_2[:,0], color = color[1], label = f'$h = 0.025$', lw = 2)
ax.loglog(time_3, data_3[:,0], color = color[2], label = f'$h = 0.016$', lw = 2)

ax.set_xlabel('time')
ax.set_ylabel('$u_{EE}$')

ax.set_title('Approximated solutions ')
ax.legend()

```




    <matplotlib.legend.Legend at 0x7fc9f80f40d0>




    
![png](./tutorial_ode_42_1.png)
    


**Answer :**

The above differential equation is an example of a [stiff equation](http://en.wikipedia.org/wiki/Stiff_equation), whose solution can be written as the sum of a fast and a slow component.

**Question.** Determine the closed form of the solution and identify its respective fast and slow components.

**Answer.**

The closest form is the third one, the step is small enough to detect strong variations imposed by the first term in the differential equation, contrasting with the first solution. The slow component is the $\cos t$

From a numerical point of view, one calls *stiff* any system of ordinary differential equations for which an explicit numerical solution method presents problems of numerical instability, unless the discretisation step length is extremely small (or, to be more precise, much smaller than would be required in principle for the accuracy required when using a step adaptation mechanism). The simulations previously performed indicate that we are here faced by such an equation.

**Question :** Perform the previous calculations using this time the implicit Euler method. Conclude.


```python

def f(t,u): return -50*(u-np.sin(t))+np.cos(t)

u0=np.array([1.])

time_1 = np.linspace(0,2, 40) 
data_1 = implicit_EE(f, 2, 0, u0, 40)

time_2 = np.linspace(0,2, 80) 
data_2 = implicit_EE(f, 2, 0, u0, 80)

time_3 = np.linspace(0,2, 120) 
data_3 = implicit_EE(f, 2, 0, u0, 120)

fig, ax = plt.subplots()
color = color_list(3)

    
ax.loglog(time_1, data_1[:,0], color = color[0], label = f'$h =  0.05$', lw = 2)
ax.loglog(time_2, data_2[:,0], color = color[1], label = f'$h = 0.025$', lw = 2)
ax.loglog(time_3, data_3[:,0], color = color[2], label = f'$h = 0.016$', lw = 2)

ax.set_xlabel('time')
ax.set_ylabel('$u_{EE}$')

ax.set_title('Approximated solutions ')
ax.legend()

```




    <matplotlib.legend.Legend at 0x7fc9f6631c90>




    
![png](./tutorial_ode_48_1.png)
    


**Answer :** Here we can see that the step size is no more essential for the integration stability. Indeed the fast component seems to be taken into account at low discretization. 

## Exercise 5. A chaotic dynamical system: the Lorenz system.

We are interested in [the Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system), first studied by the mathematician and meteorologist Edward Lorenz to model the atmospheric convection. It is defined as the following system of differential equations:
{{< math >}}
$$
\begin{cases}
x'(t)= \sigma(y(t) - x(t))\\
y'(t)= x(t)(\rho - z(t)) - y(t)\\
z'(t)= x(t)y(t) - \beta z(t)
\end{cases},\ t>0,
$$
{{< /math >}}
completed by initial conditions, where $x, y, z : \mathbb{R} \rightarrow \mathbb{R}$ and the real numbers $\sigma$, $\rho$ and $\beta$ are fixed numerical parameters. In his experiments, Lorenz used the following values for these parameters: $\rho=28$, $\sigma=10$ and $\beta=\frac{8}{3}$.


```python
rho = 28.
sigma = 10.
b = 8./3.
```


```python

def lorentz_equation(time, data) : 
    
    x = data[0]
    y = data[1]
    z = data[2]
    
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - b * z
    
    return np.array([dx, dy, dz])
```

We want to numerically solve the resulting system using a two-stage [Runge--Kutta method](https://en.wikipedia.org/wiki/Runge–Kutta_methods), with Butcher's tableau
{{< math >}}
$$
\begin{array}{c|cc}
0&&\\
1&1&\\
\hline
&\frac{1}{2}&\frac{1}{2}
\end{array},
$$
{{< /math >}}
called [Heun's method](https://en.wikipedia.org/wiki/Heun%27s_method).

**Question.** Show theoretically that Heun's method is a second-order method.

**Answer.** With Heun's method we have : 

$\tilde{y}_{i+1} = y_i + h f(t_i,y_i)$
{{< math >}}
$y_{i+1} = y_i + \frac{h}{2}[f(t_i, y_i) + f(t_{i+1},\tilde{y}_{i+1})]$
{{< /math >}}
if we develop the truncated error expression near $y_{i}^{true}$ we found that the truncation error scales like : 
$h^2 $ (linear terms being eliminated by the difference)

**Question.** On the model of the previous functions, write an independent function named `rk2` implementing Heun's method.


```python
def rk2(f, final_time, init_time, init_data, N_steps) : 
    
    integrated_data = np.zeros(shape = (N_steps, len(init_data)))
    integrated_data[0] = init_data
    h = (final_time - init_time) / N_steps
    
    for i in range(1, N_steps) : 
        
        time = i*h
        
        k1 = f(time, integrated_data[i-1])
        k2 = f(time + h, integrated_data[i-1] + h * k1)

        
        coeffs = np.array([1/2 * k1, 1/2* k2])        
        Butcheur_sum = np.sum(coeffs, axis = 0)
        integrated_data[i] = integrated_data[i-1] + h * Butcheur_sum
        
    return integrated_data
```

**Question.** Solve numerically the Lorenz system on the interval $[0,T]$ using Heun's method with the following initial values: $x (0) = y (0) = z (0) = 1$. We will use at least $T = 100$ and $N = 10000$.


```python
u0=np.array([1.,1.,1])

data = rk2(lorentz_equation, 100, 0, u0, 10**5)
parameter_name = ['x', 'y', 'z']
color = color_list(3)

```

**Question.** Plot the solution in the 3D-space $(x,y,z)$. The ``mplot3d`` toolkit can be used for generating 3D plots (see a [tutorial](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)).


```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = data[:,0]
y = data[:,1]
z = data[:,2]

ax.plot(x,y,z, color = color[2], lw = .5)
```




    [<mpl_toolkits.mplot3d.art3d.Line3D at 0x7fc9f82ec810>]



    /home/andrea/anaconda3/lib/python3.11/site-packages/IPython/core/pylabtools.py:152: UserWarning: constrained_layout not applied because axes sizes collapsed to zero.  Try making figure larger or axes decorations smaller.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](./tutorial_ode_60_2.png)
    


This figure is characteristic of the *Lorenz attractor*. For some specific values of its parameters, as those chosen here, the system is chaotic: this means that solutions are highly senstive to initial conditions (small changes in the initial conditions produce large changes in the solutions).

**Question.** For the same values of the parameters, slightly change the initial values and observe the chaotic behaviour of the solution.


```python
u0=np.array([2.,1.,1])

data2 = rk2(lorentz_equation, 100, 0, u0, 10**5)
parameter_name = ['x', 'y', 'z']

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x2= data2[:,0]
y2 = data2[:,1]
z2 = data2[:,2]

ax.plot(x,y,z, color = color[2], lw = .3)
ax.plot(x2,y2,z2, color = color[1], lw = .3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Chaotic behavior of Lorentz system')



```




    Text(0.5, 0.92, 'Chaotic behavior of Lorentz system')




    
![png](./tutorial_ode_62_1.png)
    


**Answer.**

## Exercise 6. An Adams-Bashforth method.
When using one previous approximate value to compute the next one in a one-step method, the only way to improve the accuracy is to resort to a [Runge-Kutta method](https://en.wikipedia.org/wiki/Runge–Kutta_methods) with more stages. In [linear multistep methods](https://en.wikipedia.org/wiki/Linear_multistep_method), accuracy is improved by using more previous approximate values, that is using more than one step. We lok at an exemple of such a method in this exercise.

We consider the use of the (explicit) three-step Adams-Bashforth method, defined by
{{< math >}}
$$
u^{(n+1)}=u^{(n)}+\frac{h}{12}\left(23f(t_n,u^{(n)})-16f(t_{n-1},u^{(n-1)})+5f(t_{n-2},u^{(n−2)})\right),
$$
{{< /math >}}
for the solution to the Cauchy problem
{{< math >}}
$$
\begin{cases}
u'(t)=f(t,u(t)),\ t>0,\\
u(0) = u_0.
\end{cases}
$$
{{< /math >}}

**Question.** Using the integral form of the ordinary differential equation and considering steps of equal size $h$, show how the coefficients of the method are obtained using Lagrange polynomial interpolation of the right-hand side at the points $t_{n-2}$, $t_{n-1}$ and $t_n$.

Soit z une solution exacte de l'équation. On a alors d'après Jean-Pierre Demailly, *Analyse numérique et équations différentielle*:
{{< math >}}
$z(t_{n+1})=z(t_{n})+\int _{t_{n}}^{t_{n+1}}f(t,y)\,\mathrm {d} t.$
{{< /math >}}
Supposons que les points $(z(t_{n-i}))$ et les pentes {{< math >}}$(f_{n-i})=(f(t_{n-i},z(t_{n-i})))${{< /math >}} soient connues pour {{< math >}}$0≤ i≤r${{< /math >}}. 
as
On calcule alors le polynôme d'interpolation de Lagrange de ces points :
{{< math >}}
$P_{n,r}(t)=\sum _{i=0}^{r}f_{n-i}L_{n,i,r}(t),$
{{< /math >}}
avec les polynômes de Lagrange suivants
{{< math >}}
$L_{n,i,r}(t)=\prod _{0\leq j\leq r,j\neq i}{\frac {t-t_{n-j}}{t_{n-i}-t_{n-j}}}.$
{{< /math >}}
On fait alors l'approximation :

{{< math >}}
$z(t_{n+1})\simeq z(t_{n})+\int _{t_{n}}^{t_{n+1}}\sum _{i=0}^{r}f_{n-i}L_{n,i,r}(t)\,\mathrm {d} t=z(t_{n})+\sum _{i=0}^{r}f_{n-i}\int _{t_{n}}^{t_{n+1}}L_{n,i,r}(t)\,\mathrm {d} t.$

{{< /math >}}
La méthode d'Adams-Bashforth à r+1 pas s'écrit donc :
{{< math >}}
${\begin{cases}y_{n+1}&=y_{n}+\Delta t_{n}\sum _{i=0}^{r}f_{n-i}b_{n,i,r}\\t_{n+1}&=t_{n}+\Delta t_{n}\\f_{n+1}&=f(t_{n+1},y_{n+1}).\end{cases}}${{< /math >}}
avec
{{< math >}}
$b_{n,i,r}={\frac {1}{t_{n+1}-t_{n}}}\int _{t_{n}}^{t_{n+1}}L_{n,i,r}(t)\,\mathrm {d} t.$
{{< /math >}}
Il vient alors directement les coefficients voulus 

**Question.** What is the order of the method?



**Answer.** FOr this type of Adams-Bashforth method, the order of convergence is 3 since the truncation error verify : 
{{< math >}}
$|\tau_{n}|=\left|z(t_{n+1})-y_{n+1}\right|\leq \max _{[t_{0},t_{0}+T]}|z^{(r+2)}|\cdot \Delta t_{n}\cdot \max _{n}(\Delta t_{n})^{r+1}${{< /math >}}

**Question.** Write, on the model of the previous functions, a function named `ab3` implementing the three-step Adams-Bashforth method. Explain how the method is effectively started from the single value $u_0$.

**Answer.**


```python
def ab3(f, final_time, init_time, init_data, N_steps) :
    
    integrated_data = np.zeros(shape = (N_steps, len(init_data)))
    
    h = (final_time - init_time) / N_steps
    
    init_data_rk = rk4(f, 2*h + init_time, init_time, init_data, 3)
    integrated_data[:3] = init_data_rk
    
    for i in range(3, N_steps) : 
        
        time = i*h
        
        integrated_data[i] = integrated_data[i-1] + h / 12 * (23 * f(time, integrated_data[i-1]) - 16 * f(time - h, integrated_data[i - 2]) + 5 * f(time - 2*h, integrated_data[i - 3]))
        
    return integrated_data
        
```

**Question.** Determine numerically the order of the method by solving a problem for which the solution is explicitely known.

**Answer.**


```python
data = ab3(f, 14, 0, u0, 10**5)

parameter_name = ['S', 'I', 'R']
fig, ax = plt.subplots()
color = color_list(3)
u0=np.array([762.,1.,0])
time = np.linspace(0,14, 10**5)

for k in range(len(u0)) : 
    
    ax.plot(time, data[:,k], color = color[k], label = parameter_name[k])
    ax.set_xlabel('time')
    ax.set_ylabel('Population')
    
    ax.set_title('Results of ab3 integration')
    ax.legend()
```


    
![png](./tutorial_ode_73_0.png)
    



```python
N = np.arange(10,10**4, 10)
all_predicted_data =[]
errors = []

for Num_steps in N :
    
    all_predicted_data.append(ab3(f, 14, 0, u0, Num_steps))
    errors.append(np.linalg.norm(data[-1] - all_predicted_data[-1][-1],2))
    
tau = 1/N
fig, ax = plt.subplots()
ax.plot(np.log(tau), np.log(errors), 'o', color = color[1], markersize = 3)
fit = np.polyfit(np.log(tau), np.log(errors), deg  = 1)
poly = np.poly1d(fit)

ax.plot(np.log(tau), poly(np.log(tau)), '--', color = color[2], label = f'fit  : y = {fit[0] : .2f} x + {fit[1]: .2f}')
ax.set_xlabel('power of discretization')
ax.set_ylabel('errors')
ax.legend()
ax.set_title('Order of Convergence')
```




    Text(0.5, 1.0, 'Order of Convergence')




    
![png](./tutorial_ode_74_1.png)
    

