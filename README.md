## LSTM from Scratch

### LSTM mechanism
* x_t : input vector shape 
* f_t : forget gate's activation vector value between 0 and 1
* i_t : input/update gate's activation  vector value between 0 and 1
* o_t : output gate's activation vectors vector value between 0 and 1

* h_t : hiddent state vectors known as output vector of LSTM the value of vectors between -1 and +1 

* c_prime_t : cell input activation vector 
* c_t ~ cell activator


Activation function 

* sigg: sigmoid function
* sigc: hyperbolic tagent function
* sigh: hyperbolic tagent or, as peephole LSTM 

* @  ~ dot product
* * elementvise dot product 


f_t = sigg(W_{f} @ x_{t} + Uf @ h_{t-1} + bf)
i_t = sigg(W_{i} @ x_{t} + Ui @ h_{t-1} + bi)
o_t = sigg(W_{o} @ x_{t} + Uo @ h_{t-1}+ bo)

c_til_{t} = singc(W_{o} @ x_{t} + Uc @ h_{t-1} + bc)

ct = f_t * c_{t-1} + i_t * c_til_{t}
ht = o_t * sigh(ct)  


### The idea  when information flow to the cell

* so every current input and previous hidden state will flow pararelly to three linear layers to neural network which they use singmoid function to control which path that allow information to flow. There are 3 known as forget, input/update and output gate. On the other hand, cell state are get information flow from previous hidden state and current as well , but  using hyperbolic tangent instead.
  
  * o_t : output gate  are used to control ouput like how much that allow  information from current state to be output
  * i_t : update gate how much to keep information of current state
  * f_t : how much previouse state information to flow or pass
  * c_til_{t} : compute current information based on previous hidden state and current input it might get new pattern from this computation.
  
  * c_t : cell state collect both pattern and old pattern which depend on update gate and forget allow information to pass  

* so previous cell state would be control by forget while c_til_{t} which control by input/update gate.    



### Todo 

   1. test foward one directional
   2. create bidirectional LSTM
