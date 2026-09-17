# IT Blackout Workshop - Neural Networks:

---

---

# What we will see :

1. ***Artificial Intelligence, Machine Learning & Deep Learning***
2. ***A closer look at Machine Learning***
3. ***From Logistic Regression to a Single Neuron***
4. ***How a Model Learns***
    - *Forward Propagation*
    - *Loss*
    - *Backpropagation*
    - *Gradient Descent*
    - *Training Loop*
5. ***Evaluation***
6. ***From One Neuron to an MLP***
7. ***How an MLP Learns***
8. ***Hands-on: Training an MLP on the Iris Dataset***
9. ***Conclusion***

# 1. Introduction

#### what is Artificial Intelligence ( AI ) ?

AI is when machines are built to think, work, and solve problems in a way similar to humans.

#### BUUT how can we get a machine to “think” !

That’s where the Machine Learning ( ML ) comes in…. 

ML is a field of AI that allows computers to learn patterns from data instead of being explicitly programmed for every situation

#### AND when patterns get more complex to learn ?

Here we go deeper. That’s what we call Deep Learning. It is a way of learning that uses neural networks.

#### Wait wait, what is a neural network ?

Let’s start with something you already know: YOUR BRAIN !

Your brain is made up of millions of neurons, constantly passing signals to each other. When you see a flower, no single neuron understands what a rose is ! The recognition emerges from many neurons firing together, each contributing a small piece.

Now, let’s go back to artificial neural networks. They borrow the same idea, but in mathematical form. Instead of biological neurons, we have artificial neurons. They are small units who take inputs, do calculations and pass a signal forward.

 

#### Seems complicated isn’t it ? Don’t worry we will see how this actually works, starting with a single neuron !

So, in this workshop we will:

- Start with a single neuron, and see how it makes a decision.
- Build it up into a full neural network.
- Understand how it learns from its mistakes ( Loss, Back propagation, Gradient descent )
- Watch it train, live, on real data

#### NOTE:

For AI, ML or Deep Learning, we will be coding in python ( because it is simple, open-source and has powerful libraries for data and math), using some important libraries:

- Pandas: For loading and manipulating data
- Numpy: For numerical operations and matrix computations
- Scikit-learn: For data preparation tools and for comparing our results with a ready-made model
- Matplotlib and seaborn for visualizing our data and results

---

---

# 2. Machine Learning - a closer look :

Let’s zoom in on ML for a moment before diving into neural networks. Because ML is not just one thing! 

Also, mathematically, we can describe it as a mapping: F: x —> y 

from a training set: D = { ( x(1),y(1) ), …, ( x(n),y(n) ) }

In plain words: We have n examples, each with some input features x and a correct output y ( also called a target ). The goal of training is to find a function F that, given a new x, correctly predicts the target y based on the patterns it learned from these examples.

There are several types, depending on how machine learns.

In this workshop we are focusing on  Supervised Learning and more precisely: The Classification Task !

Within, there is a typical pipeline we follow to build every ML model:

1. Data preparation
2. Model training ( This workshop focuses entirely on this step where we will see Forward propagation, Loss, Back propagation, Optimization )
3. Evaluation

#### WAIIIT ! Before moving forward, a quick but important note:

Data preparation deserves real respect here. There is a well-known saying in ML : “ Garbage in, garbage out ”. So, no matter how powerful your model is, if the data you feed it is messy, incomplete, or poorly structured, your model will learn poorly too !

  

Soo, without going deep, data preparation typically includes: 

1. Loading data: Using pandas ( pd.read_csv.(path) )
2. Cleaning data: Encoding categorical data ( One-hot encoding, Binary encoding, Label encoding), Feature selection, Handling missing values, Data scaling ( Normalization, Standardization), Feature engineering. They are mostly done using pandas and scikit-learn (sklearn.preprocessing)
3. Splitting our data: Dividing it into training set and testing set. Using (sklearn.model_selection.train_test_split)

We'll actually use some of these tools ourselves later, when we prepare our Iris dataset before training.

#### Training and Testing sets ( splitting ):

After the data preparation phase ( loading data, clean it, encode it, …etc), we split our data into 2 sets: training set and testing set ( mostly we give 70% or 80% to training set, the rest we keep it for testing )

Training set: The data the model learns from. It sees both the inputs (x) and correct outputs (y) and uses them to adjust its internal parameters, this is where all the learning happens.

Testing set: Unseen data used afterward to check how well the model actually performs.

#### Before moving to another thing, what is the purpose of splitting the data into testing and training ?

If we trained AND tested on the same data, the model could simply memorize the answers instead of actually learning to generalize. Testing on unseen data tells us whether the model truly learned the patterns or it  just memorized the examples it was given !

---

---

# 3. Logistic Regression - Our first neuron model:

Before jumping into a full neural network, let's start small, with just one neuron, so we can really understand what's happening before scaling up.

#### So, how does one neuron make a decision?

As we said before, in this workshop we are working with the supervised learning model and specifically THE CLASSIFICATION TASK !

We said that it is a method where the model learns how to predict categorical outputs from training on labeled data 

#### BUT, How exactly does it work ?

Mathematically, Logistic Regression computes a linear combination of the inputs: z = w·x + b ( we will go into it deeper in training phase )

This is a linear function, geometrically, it defines a straight line (in 2D) or a flat plane (in higher dimensions).

#### Let’s start training now !

#### How do we separate groups ( categories ) ?

It starts with a simple calculation: **Z = w *  X +b** 

this formula is responsible for drawing the line ( called also the decision boundary ) by going through the steps;

#### Step 1 - Weighted sum:

 The neuron takes each input **X**, multiplies it by the weight **w**  ( a number representing how important that input is ),  ****and adds a bias. This gives us **Z** which is a raw value.

PS: Initially, we choose **w** randomly and give **b** a value of zero !

BUUT,

The problem is that: **Z** could be any number ( huge, tiny, negative, …etc) WHILE we need a probability ( we need it to be between 0 and 1 )

SOOO,

We found a solution which is: a function that takes any number and squashes it into the range [ 0 ; 1 ] so our output could be a probability

That’s what we will see in the step 2 !

#### Step 2 - Sigmoid function:

This “ squashing function “ is actually called Sigmoid function.

**sigmoid(Z) = 1 / (1 + e^(-Z))**

Sigmoid takes our raw value **Z** and transforms it into a number **a** between 0 and 1 which is our probability. 

for example: **a = 0.87** means “ **87%** chance this belongs to class 1 “

#### Step 3 - Make decision:

Now, we have our probability “ **a** “, we have to take a real decision ( **X** is in the class 1 or 0 ) 

Here, we apply a threshold ( generally 0.5) and we compare it: if a>0.5 then we put it in class 1, otherwise, class 0

#### NOTE:

The steps: weighted sum, sigmoid function and make decision all together are called Forward Propagation !

#### Step 4 - Loss:

Now that we have the prediction **a** ( ex: 0.87 ) and we already know the target **Y** ( ex: 1 is the class ), we have to measure : “how wrong we were ?”

THIIIIIS IS THE LOSS !

This measure has a name: Binary Cross-Entropy.

**Loss = -[ y × log(a) + (1 - y) × log(1 - a) ]**

since we have ( in our example ) **Y=1** and **a= 0.87** we will find 

**Loss = 0.14** # This is a small loss cause our prediction was close to the true answer. 

But, if our prediction was for ex: **a=0.1** here the loss is much higher **Loss = 2.3** so the prediction is false 

**THE WORSE THE PREDICTION, THE HIGHER THE LOSS !**

Okay, now we know how wrong we were thanks to the LOSS. But, that’s not enough because we need to know how to adjust **w** and **b** to make the loss smaller leading to better predictions.

That’s exactly what THE GRADIENT tells us ! :  a number that shows how much the loss would change if we nudged **w** or **b** a tiny bit and in which direction.

#### Soo, how do we actually calculate the gradient?

#### That’s exactly what we will do in the next step

#### Step 5 - Backpropagation:

The process we use to calculate this gradient is called BACKPROPAGATION !

For our single neuron, the math simplifies:

**∂Loss/∂w = (a - y) × X
∂Loss/∂b = (a - y)**

Let’s use our example: **a=0.87**, **Y=1**, and let’s say **X=2**

so : 

∂Loss/∂w = (0.87 - 1) × 2 = **-0.26**
∂Loss/∂b = (0.87 - 1) = **-0.13**

Both gradients are negative, this tells us we should increase **w** and **b** slightly to reduce the loss next time !

#### Step 6 - Gradient Descent:

After the calculation of the gradients (`∂Loss/∂w = -0.26`, `∂Loss/∂b = -0.13`) Now, we can adjust **w** and **b** using these formulas :

**w = w - α × (∂Loss/∂w)                                                                                                                                             b = b - α × (∂Loss/∂b)**

where the α is the learning rate : it controls how big each adjustment step is. We choose it manually. But if it is  too small: means the model will learn slowly. If it is too large, the model will never reach the best values !

Let's finish our example. Say **w** was initially **0.5**, and we use a learning rate **α = 0.1**.

w_new = w - α × (∂Loss/∂w)
w_new = 0.5 - 0.1 × (-0.26)
**w_new = 0.526**

Same for **b**, say it started at **0**:

b_new = b - α × (∂Loss/∂b)
b_new = 0 - 0.1 × (-0.13)
**b_new = 0.013**

Small steps, but every time we repeat this cycle, **w** and **b** get a little bit closer to values that make good predictions !

![gradient_descent.png](gradient_descent.png)

#### Step 7 - Training Loop:

Now let's put it all together. We don't just do this cycle once, we repeat it, over and over:

Forward Propagation → Loss → Backpropagation → Gradient Descent → repeat

Each full pass through this cycle is called an epoch. With every epoch, **w** and **b** get a tiny bit better, and the loss gets a tiny bit smaller.

After enough epochs (maybe 100, maybe 1000), the loss becomes small and stable and our single neuron has learned to make good predictions!

And that's it — that's the entire training process for a single neuron! 

Let's actually see this in action. Here's the decision boundary right at the start, with random **w** and **b**:

![boundary_before.png](boundary_before.png)

Notice there's no meaningful separation yet, the weights haven't learned anything.

Now here's the same model after training (after repeating Steps 1 through 6 many times):

![boundary_after.png](boundary_after.png)

The line has moved into place it now correctly separates Class 0 from Class 1. This is the real decision boundary we talked about. 

after this, our model will be ready for the third and last step, which is Evaluation ( but we will talk about it brievly since it is not our main topic ) which has the same steps whether we are using one neuron or a full network with multiple neurons !

---

---

# 4. Evaluation:

Now, that our model is trained, it's time to check how well it actually performs.

This is where we use our testing set: The data the model has never seen during training.

The most common metric is Accuracy:

**Accuracy = (number of correct predictions) / (total number of predictions)**

For example, if our model correctly classifies 27 out of 30 test flowers, that's an accuracy of 90%.

But accuracy alone doesn't always tell the full story. That's why we also use a Confusion Matrix: A table showing exactly which classes the model confused with which. It helps us see not just how often the model is wrong, but how it's wrong.

![confusion_matrix.png](confusion_matrix.png)

We won't dive deeper into Evaluation today, since our focus is Training, but you'll see Accuracy used shortly in the example code.

---

---

#### Now, let's see what happens when we connect many of these neurons together !

#### First of all: Why do we need to connect multiple neurons ?

A single neuron can only draw a straight line to separate 2 classes. That works well when your data is simple but what if the categories can’t be separated by a straight line at all ? like what if the shape of our data is not linear ? 

This is where the limit of one neuron becomes clear. The solution is MLP (Multi-Layer Perceptron) and the breakthrough that made it practical to train came in the 1980s, with the popularization of the backpropagation algorithm.

---

---

# 5. Multi-Layer Perceptron (MLP):

#### So, what exactly is an MLP ?

MLP is a neural network made of multiple neurons, organized into layers:

- An input layer
- One or more hidden layers
- An output layer

Each neuron in one layer is connected to every neuron in the next layer.

![archi_mlp.png](archi_mlp.png)

Unlike a single neuron, which outputs one probability (perfect for 2 classes), an MLP can have multiple neurons in its output layer. This means MLP can naturally handle problems with more than 2 categories, not just binary classification.

Mathematically, instead of a single  **Z = w * X + b**, we, now, have many of these calculations happening in parallel, layer after layer.

This is exactly what we meant by Deep Learning, back in the introduction: instead of relying on a single neuron, we go "deeper" by stacking multiple layers of neurons together and MLP is the simplest example of this idea in action !

#### SOO, how does it work ?

Just like with our single neuron, we have 4 important steps: Forward Propagation, Loss, Back Propagation and Gradient Descent.

#### Step 1 - Forward propagation:

It follows the same principle as with one neuron, but now, instead of a single weight **w**, each layer has many weights, organized into what we call a weight matrix, **W**.

Why? Because we now have multiple neurons in each layer, and every single neuron needs its own set of weights, one weight per input it receives.

For example, if a layer has **n** inputs and **m** neurons, each of those m neurons needs its own n weights. That means the weight matrix **W** for that layer holds n × m numbers in total, not just one.

The formula stays exactly the same in shape as before, just using this matrix instead of a single number:

**Z1 = X · W1 + b1**

This single line calculates the raw value for every neuron in the layer at once, thanks to matrix multiplication, we don't need to compute each neuron one by one, no matter how many neurons there are.

That gives us a linear result, but we don't want to stop there! So we pass it through an activation function (ReLU) to remove the linearity:

**A1 = ReLU(Z1)
ReLU(z) = max(0, z)**

Now, **A1** (the output of this layer) becomes the input of the next layer, using the same idea, another weight matrix or matrixes until we reach the output layer. Here we follow also the same idea: the weight matrix **W2** . ( For example, let’s suppose that we have one hidden layer so we got 2 matrixes )

PS: in another examples you can find more than 1 layer so more matrixes !

**Z2 = A1 · W2 + b2**
**A2 = softmax(Z2)
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)**

Here, we use a different activation function: Softmax, instead of Sigmoid. Why? Because Sigmoid gives us one probability (yes/no, class 0 or 1), but what if we have more than 2 possible classes?

Softmax gives us a probability for each class, all summing up to 1, for example, with 3 classes: (0.05, 0.10, 0.85).

And that's it! That's Forward Propagation for an MLP: data flows through each layer, gets transformed by weight matrices and activation functions, until we reach a final prediction.

![forwardprop_mlp.png](forwardprop_mlp.png)

#### Step 2 - Loss:

Just like before, once we have a prediction (**A2**), we need to measure how wrong it was. That's still the Loss.

The only difference: since we now have more than 2 classes instead of just 2, we use Categorical Cross-Entropy instead of Binary Cross-Entropy:

**Loss = -Σ yᵢ × log(aᵢ)**

Same idea as before: The worse the prediction, the higher the loss. We won't re-explain it in detail here since the principle is exactly the same as what we saw with a single neuron.

#### Step 3  - Backpropagation:

Remember how, in Forward Propagation, information travels from input to output? Backpropagation is the opposite: the ERROR travels from output back to input. That's exactly why it's called "back"-propagation.

Here's how it works:

First, we look at the output layer. Once we have our prediction and we know the true answer, we can easily calculate how wrong each output neuron was, exactly like we did with a single neuron.

Then, we move backward to the hidden layer. Here's the question: how much did each hidden neuron contribute to the mistake? These neurons didn't make a direct prediction, they just passed information forward. So how do we assign them blame?

The idea: Each hidden neuron was connected to every output neuron, through weights. If a hidden neuron had a strong connection to an output neuron that made a big mistake, it carries a bigger share of the blame. If the connection was weak, it carries less blame. So we redistribute the error backward, proportional to the strength of each connection.

One exception: If a hidden neuron had been "turned off" by ReLU (its value was 0), it had zero influence on the final prediction so it gets zero blame too, no matter how strong its connections were.

Once every neuron knows its share of the blame, we can calculate exactly how much each individual weight in the network contributed to the error, and that's what lets us adjust every single weight afterward, using Gradient Descent.

![backprop_mlp.png](backprop_mlp.png)

#### Step 4 - Gradient Descent:

Just like before, now that we have the gradients **(dW1, db1, dW2, db2)**, we use them to update all the weights and biases:

**W1 = W1 - α × dW1
b1 = b1 - α × db1
W2 = W2 - α × dW2
b2 = b2 - α × db2**

Same formula, same idea as with a single neuron. The only difference is that now we're updating entire matrices of weights at once, instead of just one number.

And that's it ! Every single weight in the network takes a tiny step in the right direction, making the network's next prediction a little bit better.

#### AND FINALLY - Training Loop:

Just like before, we don't do this cycle just once, we repeat it, over and over:

Forward Propagation → Loss → Backpropagation → Gradient Descent → repeat

Each full pass through this cycle is still called an epoch. With every epoch, all the weights **(W1, b1, W2, b2)** get a tiny bit better, and the loss gets a tiny bit smaller.

BUUT, we cant just add more and more epochs because it can cause overfitiing ( our model will memorize instead of generalize ) 
So, we just use a technique called early stopping. It stops the training when acuuracy is no more changing or becomes hardly changing 

After enough epochs, the loss becomes small and stable, and our MLP has learned to classify data it has never seen before.

And that's it ! that's the entire training process for a full neural network!

After the training we will do the evaluation which is exactly the same as for one neuron !

![datashape_mlp.png](datashape_mlp.png)

---

---

```python
# 6.CODE:
IMPORTING LIBRARIES:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
1. LOADING DATA:
----------------
df = pd.read_csv (r"C:\VSCode\Jupyter\Workshop\Iris\iris.csv")
print (df.head())
df.info()
#for searching about NULL values we can also use this methode:
print ( df.isnull().sum())
print ( df["species"].unique())
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
2. SHUFFLE DATA:
----------------
df_sh = shuffle (df, random_state = 42).reset_index(drop=True)
df_sh
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
3. ENCODE TARGET:
-----------------
mapping =  {
            "setosa" : 0,
            "versicolor" : 1,
            "virginica" : 2

            }
Y = df_sh ["species"].map(mapping)
X = df_sh.drop (["species"], axis = 1)
print (X.head())
print (Y.head())
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
4. TRAIN/TEST SPLIT:
---------------------
X_train, X_test, Y_train, Y_test = train_test_split (
    X, Y, test_size = 0.2, random_state = 42
)
print ( X_test.shape)
print ( X_train.shape)
print (Y_test.shape)
print (Y_train.shape)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
5. STANDARDIZATION:
-----------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform (X_train)
X_test_sc = scaler.transform (X_test)
print (X_test_sc)
print ( X_train_sc)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
6. NEURAL NETWORK:
---------------------
model = MLPClassifier(
    hidden_layer_sizes = (5,),
    activation = 'relu',
    solver = 'sgd', # SGD = the Gradient Descent algorithm we built from scratch (w = w - α × gradient)
    learning_rate_init = 0.1, # this parameters will be changed due to experiements
    max_iter = 1000, #The model can stop earlier if it considers that it has converged!
    random_state = 42
)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
7. TRAINING:
------------
model.fit ( X_train_sc, Y_train)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
8. TEST:
---------------
Y_pred = model.predict(X_test_sc)
print ("Prediction are:", Y_pred)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
9. ACCURACY: 
------------
accuracy = accuracy_score(Y_test, Y_pred)
print ( "ACCURACY:", accuracy)
print ( classification_report(Y_test, Y_pred) )
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
10. Plot Lost Curve:
--------------------------
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_, color='purple', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve During Training')
plt.grid(alpha=0.3)
plt.show()

```

---

---

# 7. Conclusion:

And that's it! Let's quickly recap.

We started with a single neuron — Logistic Regression — and saw how it makes a decision (Weighted Sum → Sigmoid → Threshold), and how it learns from its mistakes (Loss → Backpropagation → Gradient Descent), repeated over epochs.

Then, since a single neuron can only draw a straight line, we connected many neurons together into an MLP — and saw that the exact same cycle applies, just at a bigger scale, across multiple layers.

So next time you hear "the model is training," you'll know exactly what's happening:

Forward Propagation → Loss → Backpropagation → Gradient Descent → repeat.

Thanks for going through Blackout with me — you just turned a black box into something you can explain!

---

---