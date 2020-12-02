---
layout: post
title:  "Backpropagation"
date:   2017-02-24 07:00:05 -0600
categories: "Deep Learning"
---
-----
<div class="fig figcenter fighighlight">
  <img src="/figures/backpropagation/simple_nn.png" width="100%">
  <div class="figcaption" style="color:gray; font-size:16px; font-family:monospace" align="center">
        Figure-01: A simple artificial neural network.
  </div>
</div>
------
&nbsp;

I am going to use the following convention for designations of symbols.


\\(w_{ds}^{l}\\) = weight associated with the connection from neuron \\(s\\) in layer \\((l-1)\\)
to neuron \\(d\\) layer \\(l\\)



\\(b_{d}^{l}\\) = bias of neuron \\(d\\) in layer \\(l\\)


Here \\(s\\) and \\(d\\) stand for the indices of source \\((s)\\) neuron in the previous layer and destination \\((d)\\) neuron in the current layer. I pick this style from the MATLAB Neural Network toolbox documentation.

Each neuron receives output from the neurons in the previous layer. To simplify
the equations, we connect each neuron in the current layer to each of the neurons in the previous layer through the weights \\(w_{ds}^{l}\\). To remove any of the connections, we just need to make the corresponding weights equal to zero.

\\(z_{d}^{l} = \sum_{s}{w_{ds}^{l}a_{s}^{l-1} + b_{d}^{l}}\\) = linear combination of the activations/outputs of all the neurons in the previous layer to neuron \\(d\\) in layer \\(l\\). This is also called weighted input for neuron \\(d\\) in layer \\(l\\).

\\(a_{d}^{l}\\) = activation/output of neuron \\(d\\) in layer \\(l\\)


At the time when backpropagation algorithm was invented by [Rumerhart, Hinton, and Williams][backprop-original], \\(sigmoid\\) was the most popular squashing function for the activation of the neurons. So, I am going to use \\(sigmoid\\) activation here.


\\(sigmoid\\) function, \\(\sigma \left ( z \right ) = \frac{1}{1 + \exp{ \left ( -z \right ) }}\\)


So, we can write the activation/output of unit \\(d\\) in layer \\(l\\) as follows:


\\(a_{d}^{l} = \sigma \left ( z_{d}^{l} \right )\\)

or, \\(a_{d}^{l} = \frac{1}{1 + \exp{\left (-\left (\sum_{s}{w_{ds}^{l}a_{s}^{l-1} + b_{d}^{l}} \right ) \right ) }} \\)

In matrix form, weight matrix, \\(\mathbf{W^{l}}\\) \\( \equiv \\) (**#** _units in layer_ \\(l\\) ) \\(\times\\) (**#** _units in layer_ \\(l-1)\\)

For example, here in figure 1, the expression for the weighted inputs for for layer 3 can be written in matrix form as follows.

$$
\begin{align*}
\begin{bmatrix}
    z_{1}^{3} \\\ z_{2}^{3} \\\ z_{3}^{3}
\end{bmatrix}
=
\begin{bmatrix}
    w_{11}^{3} && w_{12}^{3} && w_{13}^{3} && w_{14}^{3} \\\
    w_{21}^{3} && w_{22}^{3} && w_{23}^{3} && w_{24}^{3} \\\
    w_{31}^{3} && w_{32}^{3} && w_{33}^{3} && w_{34}^{3}        
\end{bmatrix}
\begin{bmatrix}
    a_{1}^{2} \\\ a_{2}^{2} \\\ a_{3}^{2} \\\ a_{4}^{2}
\end{bmatrix}
+
\begin{bmatrix}
    b_{1}^{3} \\\ b_{2}^{3} \\\ b_{3}^{3}
\end{bmatrix}
\end{align*}
$$

\\(
\hspace{6em}(3 \times 1)
\hspace{6em} (3 \times 4)
\hspace{4.8em} (4 \times 1)
\hspace{0.9em} (3 \times 1)
\\)

$$
or, \hspace{1em} \mathbf{z^{3} = W^{3} a^{2} + b^{3}}
$$

In general, the expression for the weighted inputs for layer \\(l\\) is given by

$$
\mathbf{z^{l} = W^{l} a^{l-1} + b^{l}}
$$

_where,_ the dimensions of the vectors and matrces are

\\(\mathbf{z^{l}}\\) \\( \equiv \\) (**#** _units in layer_ \\(l\\) ) \\(\times\\) 1

\\(\mathbf{W^{l}}\\) \\( \equiv \\) (**#** _units in layer_ \\(l\\) ) \\(\times\\) (**#** _units in layer_ \\(l-1)\\)

\\(\mathbf{a^{l-1}}\\) \\( \equiv \\) (**#** _units in layer_ \\(l-1)\\) \\(\times\\) 1

\\(\mathbf{b^{l}}\\) \\( \equiv \\) (**#** _units in layer_ \\(l)\\) \\(\times\\) 1

-----
&nbsp;

Now, we are all set for the backpropagation algorithm. So, let us take a look at this algorithm from the [Rumerhart, Hinton, and Williams][backprop-original] paper.

Let,

\\( \mathbf{y} \\) = desired output vector

\\( L \\) = output layer

Total squared error, \\( E = \frac{1}{2} \sum_{c} \sum_{d} (a_{d}^{L} - y_{d})^{2} \\)

\\( c \\) = index over training cases (input-output pairs)

\\( d \\) = index over output units or neurons in the final layer

The simplest version of the [gradient descent][gradient-descent-ng] algorithm for iterative weight updates to minimize the error \\( (E) \\) is given by the following equation.

\begin{equation}
w_{ds}^{l}(t+1) = w_{ds}^{l}(t) + \Delta w_{ds}^{l}(t) \\\
\text{where, } \Delta w_{ds}^{l}(t) = -\eta \frac{\partial E}{\partial w_{ds}^{l}}
\end{equation}

$$
\begin{align*}
\eta = \text{learning rate}
\end{align*}
$$

$$
\begin{align*}
w_{ds}^{l}(t) = \text{weight at iteration } t
\end{align*}
$$

$$
\begin{align*}
w_{ds}^{l}(t+1) = \text{weight at iteration } (t+1)
\end{align*}
$$

Therefore, to minimize \\( E \\) by gradient descent, it is necessary to compute the partial derivative of \\( E \\) with respect to each of the weights in the network. This is simply the sum of the partial derivatives for each of the input-output pairs.

Differentiating the error equation for a particular case \\( c \\) and suppressing the index c for notational simplicity gives

\begin{equation}
E = \frac{1}{2} \sum_{c} \sum_{d} (a_{d}^{L} - y_{d})^{2} \\\
E_{c} = \frac{1}{2} \sum_{d} (a_{d}^{L} - y_{d})^{2}
\end{equation}

Suppressing c for a single training case for notational simplicity,

\begin{equation}
E \equiv E_{c} = \frac{1}{2} \sum_{d} (a_{d}^{L} - y_{d})^{2}; \hspace{2em} [L = \text{final layer}]
\end{equation}

We know,
\begin{equation}
a_{d}^{L} = \sigma \left ( z_{d}^{L} \right )  \\\
z_{d}^{L} = \sum_{s}{w_{ds}^{L}a_{s}^{L-1} + b_{d}^{L}}
\end{equation}

Partial derivative w.r.t. the weights and biases in the final layer (\\( L \\)) can be determined using the chain rule.

\begin{equation}
\frac{\partial E}{\partial w_{ds}^{L}} =
\frac{\partial E}{\partial a_{d}^{L}} \;
\frac{\partial a_{d}^{L}}{\partial z_{d}^{L}} \;
\frac{\partial z_{d}^{L}}{\partial w_{ds}^{L}}  \\\
\frac{\partial E}{\partial b_{d}^{L}} =
\frac{\partial E}{\partial a_{d}^{L}} \;
\frac{\partial a_{d}^{L}}{\partial z_{d}^{L}} \;
\frac{\partial z_{d}^{L}}{\partial b_{d}^{L}}
\label{eq:pde1}
\end{equation}

Now, partial derivative w.r.t. output of the penultimate layer \\( a_{s}^{L-1} \\) can be written as

\begin{equation}
\frac{\partial E}{\partial a_{s}^{L-1}} = \sum_{d}
\frac{\partial E}{\partial a_{d}^{L}} \;
\frac{\partial a_{d}^{L}}{\partial z_{d}^{L}} \;
\frac{\partial z_{d}^{L}}{\partial a_{s}^{L-1}}   
\label{eq:pde2}
\end{equation}

Here, the summation is over all the units in layer \\( L \\). I think this equation as the most crucial part of the whole algorithm. The reason behind such thinking will be clearer later in this note. Now, let us compute all the constituent partial derivatives on the RHS of both the equations above.

\begin{equation}
\frac{\partial E}{\partial a_{d}^{L}} = a_{d}^{L} - y_{d} \\\
\frac{\partial a_{d}^{L}}{\partial z_{d}^{L}} = \sigma' (z_{d}^{L}) = \sigma\left (z_{d}^{L}\right ) \left ( 1 - \sigma \left (z_{d}^{L} \right ) \right )\\\
\frac{\partial z_{d}^{L}}{\partial w_{ds}^{L}} = a_{s}^{L-1} \\\
\frac{\partial z_{d}^{L}}{\partial a_{s}^{L-1}} = w_{ds}^{L}
\label{eq:pde3}
\end{equation}

We can repeat this set of computations to update the weights for successively earlier layers in the network. We are going to take a deeper look at this repeated computation shortly. At this moment, as you can see, we calculate the value of the error function by forward-propagating the values from the input layer towards the output layer. On the other hand, the learning algorithm works by adjusting the weights in each iteration by propagating the error from the output layer towards the input layer backward. This is the reason the algorithm is called __backpropagation__ (or, backward-propagation).

-----
&nbsp;

We have already seen the basic steps of backpropagation algorithm through the output layer (\\(L\\)) and its penultimate layer (\\(L-1\\)). There are some common terms in the equations (equations \ref{eq:pde1}, \ref{eq:pde2}, and \ref{eq:pde3}) of the algorithm. It would be easier to follow and implement the algorithm if we can separate the common terms in these equations from the others and thus make these equations appear simplified as well as more generalized over the layers. To do this, let us start by rewriting equations \ref{eq:pde1} and \ref{eq:pde2} for any layer \\(l\\) again below.

\begin{equation}
\frac{\partial E}{\partial w_{ds}^{l}} =
\left ( \frac{\partial E}{\partial a_{d}^{l}} \;
\frac{\partial a_{d}^{l}}{\partial z_{d}^{l}} \right ) \;
\left ( \frac{\partial z_{d}^{l}}{\partial w_{ds}^{l}}  \right )
= \delta_{d}^{l} \;  
\frac{\partial z_{d}^{l}}{\partial w_{ds}^{l}} \\\
\frac{\partial E}{\partial b_{d}^{l}} =
\left ( \frac{\partial E}{\partial a_{d}^{l}} \;
\frac{\partial a_{d}^{l}}{\partial z_{d}^{l}} \right ) \;
\left ( \frac{\partial z_{d}^{l}}{\partial b_{d}^{l}} \right )
= \delta_{d}^{l} \;  
\frac{\partial z_{d}^{l}}{\partial b_{d}^{l}} \\\
\frac{\partial E}{\partial a_{s}^{l-1}} = \sum_{d \in \left ( \text{layer } l \right )}
\left ( \frac{\partial E}{\partial a_{d}^{l}} \;
\frac{\partial a_{d}^{l}}{\partial z_{d}^{l}} \right ) \;
\left (\frac{\partial z_{d}^{l}}{\partial a_{s}^{l-1}} \right )  
= \sum_{d \in \left ( \text{layer } l \right )}
\delta_{d}^{l} \;  
\frac{\partial z_{d}^{l}}{\partial a_{s}^{l-1}} \\\
\text{where, } \delta_{d}^{l} =
\frac{\partial E}{\partial a_{d}^{l}} \;
\frac{\partial a_{d}^{l}}{\partial z_{d}^{l}}
= \frac{\partial E}{\partial z_{d}^{l}}
\label{eq:pde_mat1}
\end{equation}

So, we finally get the following set of equations.

\begin{equation}
\delta_{d}^{l}
= \frac{\partial E}{\partial z_{d}^{l}} =
\frac{\partial E}{\partial a_{d}^{l}} \;
\frac{\partial a_{d}^{l}}{\partial z_{d}^{l}} \\\
\frac{\partial E}{\partial w_{ds}^{l}}
= \delta_{d}^{l} \;  
\frac{\partial z_{d}^{l}}{\partial w_{ds}^{l}} \\\
\frac{\partial E}{\partial b_{d}^{l}}
= \delta_{d}^{l} \;  
\frac{\partial z_{d}^{l}}{\partial b_{d}^{l}} \\\
\frac{\partial E}{\partial a_{s}^{l-1}}
= \sum_{d \in \left ( \text{layer } l \right )}
\delta_{d}^{l} \;  
\frac{\partial z_{d}^{l}}{\partial a_{s}^{l-1}} \\\
\label{eq:backprop-single}
\end{equation}

Here, \\( \frac{\partial E}{\partial a_{s}^{l-1}} \\) is needed to make the error propagate backward and thus update the weights for the earlier layers. This is the reason I think the calculation of this partial derivative the most significant computational part of the algorithm.

-----
&nbsp;

All right. Now, we know the equations of backpropagation in a simple form. However, using equations \ref{eq:backprop-single}, we need to consider each of the weight updates in any layer one-by-one for each unit. It would even be easier to think about if we can compute the set of 4 partial derivatives for all the units in any layer at once in each iteration. This is possible if we can somehow express equations \ref{eq:backprop-single} for all the units in a single layer using the tricks of linear algebra. In that case, it would also be computationally more efficient since matrix computation is always faster than their iterative alternatives.  

Before going into further details, let us take a look at the [Hadamard product or Schur product][nn-deep-learning-nielsen]. This operation is simply the elementwise multiplication between two vectors or matrices. One example of this product is shown below.

$$
\begin{align*}
\begin{bmatrix}
    1 && 2 \\\ 3 && 4 \\\ 5 && 6
\end{bmatrix}
\odot
\begin{bmatrix}
    7 && 8 \\\ 9 && 10 \\\ 11 && 12
\end{bmatrix}
=
\begin{bmatrix}
    7 && 16 \\\ 27 && 40 \\\ 55 && 72
\end{bmatrix}
\end{align*}
$$

The \\( \odot \\) symbol is used to denote this product. I introduce this operation here because we need to know about this elementwise product and [matrix multiplication][matrix-multiplication] to be able to rewrite the backparagation equations in matrix form.

At this point, we will take a top-bottom approach using the network in figure 1. That is, first we will write the equations for the units in a layer one-by-one for the network in figure 1. Then we will put those equations together and try to come up with a single equation for one layer using the 2 matrix operations (Hadamard product and matrix multiplication) noted above. I add figure 1 with the same caption and number again below so that we do not need to scroll up to take a look at the network.

<div class="fig figcenter fighighlight">
  <img src="/figures/backpropagation/simple_nn.png" width="100%">
  <div class="figcaption" style="color:gray; font-size:16px; font-family:monospace" align="center">
        Figure-01: A simple artificial neural network.
  </div>
</div>
------
&nbsp;

Let us begin by writing the equation for the output layer, i.e. layer 4.

\begin{equation}
\delta_{d}^{4}
= \frac{\partial E}{\partial z_{d}^{4}} =
\frac{\partial E}{\partial a_{d}^{4}} \;
\frac{\partial a_{d}^{4}}{\partial z_{d}^{4}}
\label{eq:bp1-mat1}
\end{equation}

We have 4 units in layer 4. So, we can replace \\( d \\) by \\( \\{ 1, 2, 3, 4 \\} \\) to compute equation \ref{eq:bp1-mat1} for all the units in the layer as follows.

\begin{equation}
\delta_{1}^{4}
= \frac{\partial E}{\partial z_{1}^{4}} =
\frac{\partial E}{\partial a_{1}^{4}} \;
\frac{\partial a_{1}^{4}}{\partial z_{1}^{4}} \\\
\delta_{2}^{4}
= \frac{\partial E}{\partial z_{2}^{4}} =
\frac{\partial E}{\partial a_{2}^{4}} \;
\frac{\partial a_{2}^{4}}{\partial z_{2}^{4}} \\\
\delta_{3}^{4}
= \frac{\partial E}{\partial z_{3}^{4}} =
\frac{\partial E}{\partial a_{3}^{4}} \;
\frac{\partial a_{3}^{4}}{\partial z_{3}^{4}} \\\
\delta_{4}^{4}
= \frac{\partial E}{\partial z_{4}^{4}} =
\frac{\partial E}{\partial a_{4}^{4}} \;
\frac{\partial a_{4}^{4}}{\partial z_{4}^{4}}
\label{eq:bp1-mat2}
\end{equation}

$$
\text{or,}
\hspace{1em}
\begin{align*}
\begin{bmatrix}
    \delta_{1}^{4} \\\ \delta_{2}^{4} \\\ \delta_{3}^{4} \\\ \delta_{4}^{4}
\end{bmatrix}
=
\begin{bmatrix}
  \frac{\partial E}{\partial a_{1}^{4}} \\\
  \frac{\partial E}{\partial a_{2}^{4}} \\\
  \frac{\partial E}{\partial a_{3}^{4}} \\\
  \frac{\partial E}{\partial a_{4}^{4}}
\end{bmatrix}
\odot
\begin{bmatrix}
  \frac{\partial a_{1}^{4}}{\partial z_{1}^{4}} \\\
  \frac{\partial a_{2}^{4}}{\partial z_{2}^{4}} \\\
  \frac{\partial a_{3}^{4}}{\partial z_{3}^{4}} \\\
  \frac{\partial a_{4}^{4}}{\partial z_{4}^{4}}
\end{bmatrix}
\end{align*}
$$

$$
\begin{align}
  \text{or,} \hspace{1em}
  \mathbf{\delta^{4}} =
  \frac{\partial E}{\partial \mathbf{a^{4}}}
  \odot
  \frac{\partial \mathbf{a^{4}}}{\partial \mathbf{z^{4}}}
\label{eq:bp1-mat3}
\end{align}
$$

$$
\begin{align}
  \text{where,} \hspace{1em}
  \frac{\partial E}{\partial \mathbf{a^{4}}} =
  \begin{bmatrix}
    \frac{\partial E}{\partial a_{1}^{4}} \\\
    \frac{\partial E}{\partial a_{2}^{4}} \\\
    \frac{\partial E}{\partial a_{3}^{4}} \\\
    \frac{\partial E}{\partial a_{4}^{4}}
  \end{bmatrix}
  \hspace{1em} \text{and} \hspace{1em}
   \frac{\partial \mathbf{a^{4}}}{\partial \mathbf{z^{4}}} =
   \begin{bmatrix}
     \frac{\partial a_{1}^{4}}{\partial z_{1}^{4}} \\\
     \frac{\partial a_{2}^{4}}{\partial z_{2}^{4}} \\\
     \frac{\partial a_{3}^{4}}{\partial z_{3}^{4}} \\\
     \frac{\partial a_{4}^{4}}{\partial z_{4}^{4}}
   \end{bmatrix}
\label{eq:bp1-mat4}
\end{align}
$$

We can generalize equations \ref{eq:bp1-mat3} and \ref{eq:bp1-mat4} for any layer \\( l \\) as follows.

$$
\begin{align}
  \mathbf{\delta^{l}} =
  \frac{\partial E}{\partial \mathbf{a^{l}}}
  \odot
  \frac{\partial \mathbf{a^{l}}}{\partial \mathbf{z^{l}}}
\label{eq:bp-final-1}
\end{align}
$$

$$
\begin{align}
  \frac{\partial E}{\partial \mathbf{a^{l}}} =
  \begin{bmatrix}
    \frac{\partial E}{\partial a_{1}^{l}} \\\
    \frac{\partial E}{\partial a_{2}^{l}} \\\
    \vdots \\\
    \frac{\partial E}{\partial a_{n}^{l}}
  \end{bmatrix}
  \hspace{1em} \text{and} \hspace{1em}
   \frac{\partial \mathbf{a^{l}}}{\partial \mathbf{z^{l}}} =
   \begin{bmatrix}
     \frac{\partial a_{1}^{l}}{\partial z_{1}^{l}} \\\
     \frac{\partial a_{2}^{l}}{\partial z_{2}^{l}} \\\
     \vdots \\\
     \frac{\partial a_{n}^{l}}{\partial z_{n}^{l}}
   \end{bmatrix}
\label{eq:bp-final-2}
\end{align}
$$

$$
\begin{align*}
\text{where, } n = \text{number of units in layer } l
\end{align*}
$$

We are done with writing the first equation of backpropagation from equations \ref{eq:backprop-single} in matrix form. Now, let us write the second equation from set \ref{eq:backprop-single} for layer 4 in the network.

\begin{equation}
\frac{\partial E}{\partial w_{ds}^{4}}
= \delta_{d}^{4} \;  
\frac{\partial z_{d}^{4}}{\partial w_{ds}^{4}} \\\
\frac{\partial E}{\partial w_{1s}^{4}}
= \delta_{1}^{4} \;  
\frac{\partial z_{1}^{4}}{\partial w_{1s}^{4}} \\\
\frac{\partial E}{\partial w_{2s}^{4}}
= \delta_{2}^{4} \;  
\frac{\partial z_{2}^{4}}{\partial w_{2s}^{4}} \\\
\frac{\partial E}{\partial w_{3s}^{4}}
= \delta_{3}^{4} \;  
\frac{\partial z_{3}^{4}}{\partial w_{3s}^{4}} \\\
\frac{\partial E}{\partial w_{4s}^{4}}
= \delta_{4}^{4} \;  
\frac{\partial z_{4}^{4}}{\partial w_{4s}^{4}}
\label{eq:bp2-mat1}
\end{equation}

There are 3 units in layer \\((l-1) = 3\\). So, if we expand the above set of equations over \\( s \in \\{1,2,3\\} \\), we get

$$
\begin{align*}
  \begin{bmatrix}
    \frac{\partial E}{\partial w_{11}^{4}} &&
    \frac{\partial E}{\partial w_{12}^{4}} &&
    \frac{\partial E}{\partial w_{13}^{4}} \\\
    \frac{\partial E}{\partial w_{21}^{4}} &&
    \frac{\partial E}{\partial w_{22}^{4}} &&
    \frac{\partial E}{\partial w_{23}^{4}} \\\
    \frac{\partial E}{\partial w_{31}^{4}} &&
    \frac{\partial E}{\partial w_{32}^{4}} &&
    \frac{\partial E}{\partial w_{33}^{4}} \\\
    \frac{\partial E}{\partial w_{41}^{4}} &&
    \frac{\partial E}{\partial w_{42}^{4}} &&
    \frac{\partial E}{\partial w_{43}^{4}}
  \end{bmatrix}
  =
  \begin{bmatrix}
    \delta_{1}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{11}^{4}} &&
    \delta_{1}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{12}^{4}} &&
    \delta_{1}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{13}^{4}} \\\
    \delta_{2}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{21}^{4}} &&
    \delta_{2}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{22}^{4}} &&
    \delta_{2}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{23}^{4}} \\\
    \delta_{3}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{31}^{4}} &&
    \delta_{3}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{32}^{4}} &&
    \delta_{3}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{33}^{4}} \\\
    \delta_{4}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{41}^{4}} &&
    \delta_{4}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{42}^{4}} &&
    \delta_{4}^{4} \; \frac{\partial z_{1}^{4}}{\partial w_{43}^{4}}     
  \end{bmatrix}
\end{align*}
$$

$$
\begin{align}  
  \begin{bmatrix}
    \frac{\partial E}{\partial w_{11}^{4}} &&
    \frac{\partial E}{\partial w_{12}^{4}} &&
    \frac{\partial E}{\partial w_{13}^{4}} \\\
    \frac{\partial E}{\partial w_{21}^{4}} &&
    \frac{\partial E}{\partial w_{22}^{4}} &&
    \frac{\partial E}{\partial w_{23}^{4}} \\\
    \frac{\partial E}{\partial w_{31}^{4}} &&
    \frac{\partial E}{\partial w_{32}^{4}} &&
    \frac{\partial E}{\partial w_{33}^{4}} \\\
    \frac{\partial E}{\partial w_{41}^{4}} &&
    \frac{\partial E}{\partial w_{42}^{4}} &&
    \frac{\partial E}{\partial w_{43}^{4}}
  \end{bmatrix}
  =
  \begin{bmatrix}
    \delta_{1}^{4} && \delta_{1}^{4} && \delta_{1}^{4} \\\
    \delta_{2}^{4} && \delta_{2}^{4} && \delta_{2}^{4} \\\
    \delta_{3}^{4} && \delta_{3}^{4} && \delta_{3}^{4} \\\
    \delta_{4}^{4} && \delta_{4}^{4} && \delta_{4}^{4}
  \end{bmatrix}
  \odot
  \begin{bmatrix}
    \frac{\partial z_{1}^{4}}{\partial w_{11}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{12}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{13}^{4}} \\\
    \frac{\partial z_{1}^{4}}{\partial w_{21}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{22}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{23}^{4}} \\\
    \frac{\partial z_{1}^{4}}{\partial w_{31}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{32}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{33}^{4}} \\\
    \frac{\partial z_{1}^{4}}{\partial w_{41}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{42}^{4}} &&
    \frac{\partial z_{1}^{4}}{\partial w_{43}^{4}}     
  \end{bmatrix}  
\label{eq:bp2-mat2}  
\end{align}
$$

From equations \ref{eq:backprop-single}, we know,
\begin{equation}
z_{d}^{l} = \sum_{s} w_{ds}^{l} a_{s}^{l-1} + b_{d}^{l} \\\
\text{So,} \hspace{1em} \frac{\partial z_{d}^{l}}{\partial w_{ds}^{l}} = a_{s}^{l-1}
\hspace{1em} \text{and} \hspace{1em}
\frac{\partial z_{d}^{l}}{\partial b_{d}^{l}} = 1
 \hspace{1em} \text{and} \hspace{1em}
\frac{\partial z_{d}^{l}}{\partial a_{s}^{l-1}} = w_{ds}^{l}
\label{eq:z-derivative}
\end{equation}

So, we can write equation \ref{eq:bp2-mat2} as follows.

$$
\begin{align*}  
  \begin{bmatrix}
    \frac{\partial E}{\partial w_{11}^{4}} &&
    \frac{\partial E}{\partial w_{12}^{4}} &&
    \frac{\partial E}{\partial w_{13}^{4}} \\\
    \frac{\partial E}{\partial w_{21}^{4}} &&
    \frac{\partial E}{\partial w_{22}^{4}} &&
    \frac{\partial E}{\partial w_{23}^{4}} \\\
    \frac{\partial E}{\partial w_{31}^{4}} &&
    \frac{\partial E}{\partial w_{32}^{4}} &&
    \frac{\partial E}{\partial w_{33}^{4}} \\\
    \frac{\partial E}{\partial w_{41}^{4}} &&
    \frac{\partial E}{\partial w_{42}^{4}} &&
    \frac{\partial E}{\partial w_{43}^{4}}
  \end{bmatrix}
  =
  \begin{bmatrix}
    \delta_{1}^{4} && \delta_{1}^{4} && \delta_{1}^{4} \\\
    \delta_{2}^{4} && \delta_{2}^{4} && \delta_{2}^{4} \\\
    \delta_{3}^{4} && \delta_{3}^{4} && \delta_{3}^{4} \\\
    \delta_{4}^{4} && \delta_{4}^{4} && \delta_{4}^{4}
  \end{bmatrix}
  \odot
  \begin{bmatrix}
    a_{1}^{3} && a_{2}^{3} && a_{3}^{3} \\\
    a_{1}^{3} && a_{2}^{3} && a_{3}^{3} \\\
    a_{1}^{3} && a_{2}^{3} && a_{3}^{3} \\\
    a_{1}^{3} && a_{2}^{3} && a_{3}^{3}
  \end{bmatrix}  
\end{align*}
$$

$$
\begin{align*}
 \text{or,} \hspace{1em}
 \frac{\partial E}{\partial \mathbf{W^{4}}} =
 \begin{bmatrix}
    \delta_{1}^{4} \\\ \delta_{2}^{4} \\\ \delta_{3}^{4} \\\ \delta_{4}^{4}
 \end{bmatrix}
 \begin{bmatrix}
    a_{1}^{3} && a_{2}^{3} && a_{3}^{3}
 \end{bmatrix}
\end{align*}
$$

$$
\begin{align}
 \text{or,} \hspace{1em}
 \frac{\partial E}{\partial \mathbf{W^{4}}} =
 \mathbf{\delta^{4} \left ( a^3 \right )^{T} }
\label{eq:bp2-mat3}
\end{align}
$$

We can generalize equation \ref{eq:bp2-mat3} for any layer \\( l \\) as follows.

$$
\begin{align}
 \frac{\partial E}{\partial \mathbf{W^{l}}} =
 \mathbf{\delta^{l}  \left ( a^{l-1} \right )^{T} }
\label{eq:bp-final-3}
\end{align}
$$

We have just finished writing the second equation from equations \ref{eq:backprop-single} in matrix format. It is time to start expanding the third one for all 4 units in the output layer. Let us do that below.

$$
\begin{align*}
\frac{\partial E}{\partial b_{d}^{4}}
= \delta_{d}^{4} \;  
\frac{\partial z_{d}^{4}}{\partial b_{d}^{4}}
\end{align*}
$$

However, from equations \ref{eq:z-derivative}, we know, \\( \frac{\partial z_{d}^{l}}{\partial b_{d}^{l}} = 1 \\)

So, we can write

$$
\begin{align*}
\frac{\partial E}{\partial b_{d}^{4}}
= \delta_{d}^{4} \\\
\frac{\partial E}{\partial b_{1}^{4}}
= \delta_{1}^{4} \\\
\frac{\partial E}{\partial b_{2}^{4}}
= \delta_{2}^{4} \\\
\frac{\partial E}{\partial b_{3}^{4}}
= \delta_{3}^{4} \\\
\frac{\partial E}{\partial b_{4}^{4}}
= \delta_{4}^{4}
\end{align*}
$$

$$
\begin{align*}
\text{or,} \hspace{1em}
\begin{bmatrix}
    \frac{\partial E}{\partial b_{1}^{4}} \\\
    \frac{\partial E}{\partial b_{2}^{4}} \\\
    \frac{\partial E}{\partial b_{3}^{4}} \\\
    \frac{\partial E}{\partial b_{4}^{4}}
\end{bmatrix}
=
\begin{bmatrix}
    \delta_{1}^{4} \\\
    \delta_{2}^{4} \\\
    \delta_{3}^{4} \\\
    \delta_{4}^{4}
\end{bmatrix}
\end{align*}
$$

$$
\begin{align}
 \text{or,} \hspace{1em}
 \frac{\partial E}{\partial \mathbf{b^{4}}} =
 \mathbf{\delta^{4}}
\label{eq:bp3-mat1}
\end{align}
$$

We can generalize equation \ref{eq:bp3-mat1} for any layer \\( l \\) as follows.

$$
\begin{align}
 \text{or,} \hspace{1em}
 \frac{\partial E}{\partial \mathbf{b^{l}}} =
 \mathbf{\delta^{l}}
\label{eq:bp-final-4}
\end{align}
$$

We are done with all the matrix equations for backpropagation except the last one. Let us delve deeper into this last equation using layer \\( l = 4 \\).

$$
\begin{align*}
 \frac{\partial E}{\partial a_{s}^{3}} =
 \sum_{d=1}^{4} \delta_{d}^{4} \;
 \frac{\partial z_{d}^{4}}{\partial a_{s}^{3}}
\end{align*}
$$

However, from equations \ref{eq:z-derivative}, we know,
\\( \frac{\partial z_{d}^{l}}{\partial a_{s}^{l-1}} = w_{ds}^{l} \\)

So, we can write

$$
\begin{align*}
 \frac{\partial E}{\partial a_{s}^{3}} =
 \sum_{d=1}^{4} \delta_{d}^{4} \; w_{ds}^{4}
\end{align*}
$$

$$
\begin{align*}
\text{or,} \hspace{1em}
\begin{bmatrix}
 \frac{\partial E}{\partial a_{1}^{3}} \\\
 \frac{\partial E}{\partial a_{2}^{3}} \\\
 \frac{\partial E}{\partial a_{3}^{3}}
\end{bmatrix}
=
\begin{bmatrix}
\delta_{1}^{4} \; w_{11}^{4} + \delta_{2}^{4} \; w_{21}^{4} +
\delta_{3}^{4} \; w_{31}^{4} + \delta_{4}^{4} \; w_{41}^{4} \\\
\delta_{1}^{4} \; w_{12}^{4} + \delta_{2}^{4} \; w_{22}^{4} +
\delta_{3}^{4} \; w_{32}^{4} + \delta_{4}^{4} \; w_{42}^{4} \\\
\delta_{1}^{4} \; w_{13}^{4} + \delta_{2}^{4} \; w_{23}^{4} +
\delta_{3}^{4} \; w_{33}^{4} + \delta_{4}^{4} \; w_{43}^{4}
\end{bmatrix}
\end{align*}
$$

$$
\begin{align*}
\text{or,} \hspace{1em}
\begin{bmatrix}
 \frac{\partial E}{\partial a_{1}^{3}} \\\
 \frac{\partial E}{\partial a_{2}^{3}} \\\
 \frac{\partial E}{\partial a_{3}^{3}}
\end{bmatrix}
=
\begin{bmatrix}
w_{11}^{4} &&  w_{21}^{4} && w_{31}^{4} && w_{41}^{4} \\\
w_{12}^{4} &&  w_{22}^{4} && w_{32}^{4} && w_{42}^{4} \\\
w_{13}^{4} &&  w_{23}^{4} && w_{33}^{4} && w_{43}^{4}
\end{bmatrix}
\begin{bmatrix}
\delta_{1}^{4} \\\ \delta_{2}^{4} \\\ \delta_{3}^{4} \\\ \delta_{4}^{4}
\end{bmatrix}
\end{align*}
$$

$$
\begin{align}
\text{or,} \hspace{1em}
\frac{\partial E}{\partial \mathbf{a^{3}}}
= \mathbf{\left (W^{4} \right )^{T}} \mathbf{\delta^{4}}
\label{eq:bp4-mat1}
\end{align}
$$

We can generalize equation \ref{eq:bp4-mat1} for any layer \\( l \\) as follows.

$$
\begin{align}
\text{or,} \hspace{1em}
\frac{\partial E}{\partial \mathbf{a^{l-1}}}
= \mathbf{\left (W^{l} \right )^{T}} \mathbf{\delta^{l}}
\label{eq:bp-final-5}
\end{align}
$$

Finally, we are done. I finish this tutorial enlisting all the matrix equations \ref{eq:bp-final-1}, \ref{eq:bp-final-2}, \ref{eq:bp-final-3}, \ref{eq:bp-final-4}, \ref{eq:bp-final-5}, altogether below.

$$
\begin{align*}
\mathbf{\delta^{l}} =
\frac{\partial E}{\partial \mathbf{a^{l}}}
\odot
\frac{\partial \mathbf{a^{l}}}{\partial \mathbf{z^{l}}}
\end{align*}
$$

$$
\begin{align*}
 \frac{\partial E}{\partial \mathbf{W^{l}}} =
 \mathbf{\delta^{l}  \left ( a^{l-1} \right )^{T} }
\end{align*}
$$

$$
\begin{align*}
\frac{\partial E}{\partial \mathbf{b^{l}}} =
 \mathbf{\delta^{l}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial E}{\partial \mathbf{a^{l-1}}}
= \mathbf{\left (W^{l} \right )^{T}} \mathbf{\delta^{l}}
\label{eq:bp-final-all}
\end{align*}
$$

$$
\begin{align*}
\text{where,} \hspace{1em}
\frac{\partial E}{\partial \mathbf{a^{l}}} =
\begin{bmatrix}
  \frac{\partial E}{\partial a_{1}^{l}} \\\
  \frac{\partial E}{\partial a_{2}^{l}} \\\
  \vdots \\\
  \frac{\partial E}{\partial a_{n}^{l}}
\end{bmatrix}
\hspace{1em} \text{and} \hspace{1em}
 \frac{\partial \mathbf{a^{l}}}{\partial \mathbf{z^{l}}} =
 \begin{bmatrix}
   \frac{\partial a_{1}^{l}}{\partial z_{1}^{l}} \\\
   \frac{\partial a_{2}^{l}}{\partial z_{2}^{l}} \\\
   \vdots \\\
   \frac{\partial a_{n}^{l}}{\partial z_{n}^{l}}
 \end{bmatrix}
\end{align*}
$$

$$
\begin{align*}
\mathbf{W^{l}} =
    \begin{bmatrix}
    w_{11}^{l} && w_{12}^{l} && \dots && w_{1m}^{l} \\\
    w_{21}^{l} && w_{22}^{l} && \dots && w_{2m}^{l} \\\
    \vdots && \vdots && \dots && \vdots \\\
    w_{n1}^{l} && w_{n2}^{l} && \dots && w_{nm}^{l}
    \end{bmatrix}
\hspace{1em} \text{and} \hspace{1em}     
\mathbf{b^{l}} =
    \begin{bmatrix}
        b_{1}^{l} \\\ b_{1}^{l} \\\ \vdots \\\ b_{n}^{l}
    \end{bmatrix}    
\end{align*}
$$

$$
\begin{align*}
n = \text{number of units in layer } l
\end{align*}
$$

$$
\begin{align*}
m = \text{number of units in layer } (l-1)
\end{align*}
$$

-----
&nbsp;

#### __References__
1. [Rumelhart, Hinton, and Williams, _Learning representations by back-propagating errors_][backprop-original]
2. [Nielsen, _Neural networks and deep learning_][nn-deep-learning-nielsen]

[nn-deep-learning-nielsen]: http://neuralnetworksanddeeplearning.com/chap2.html

[backprop-original]: http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html

[gradient-descent-ng]: http://openclassroom.stanford.edu/MainFolder/VideoPage.php?course=MachineLearning&video=02.4-LinearRegressionI-GradientDescent&speed=100

[matrix-multiplication]:http://mathworld.wolfram.com/MatrixMultiplication.html
