
### 📈REPLICATING AND TESTING CUSTOM MAXOUT NETWORKS.

**Purpose: Implementing a custom Maxout network from scratch (implementing the activation function as an extention of nn.Module in Pytorch) and testing model performance on MNIST in comparison to ReLU networks.**

Hey there! This project is meant to compare the performance of two MLP models with different activation functions on the MNIST dataset. 
The first of these uses the typical ReLU activation, while the second uses **a custom maxout activation function** (replicating the original 2013 paper https://arxiv.org/abs/1302.4389) as defined in **Maxout.py**.

The objective of this experiment is to see whether the theoretical benefits of Maxout work in real-world use cases. As outlined in the above paper, Maxout can (arbitrarily) approximate any continuous function as it acts much like a piecewise linear approximator - this is promising, as it means that both the ReLU and Leaky ReLU activation functions are both special cases of Maxout. Logically, this should mean that maxout can learn more complex functions and thus be able to better model an existing relationship.

Practically, the function works by splitting the input vector into *i* groups of *k* neurons each. Then, it takes the **MAXIMUM** for each one of these groups - turning the group of *k* neurons into just one. So, if there are 10 neurons in total, and we split the layer into groups of 2 ($i=2$, $k=5$), then that means that activation function will map those 10 neurons **to just 2** - taking the **highest activations and capturing the relationships that appear to have the most relevance**.

This essentially means that **maxout also has its own weights as it acts much like a linear transform* - this produces more trainable parameters but also offers the opportunity for better learning.

**Mathematically:**
$h_1(x)=max_{jΣ[1, k]}z_{ij}$, where $z_{ij}=x^TW_{...ij}+b_{ij}$

**Visually:** (credit to @MlforNerds https://www.youtube.com/watch?v=DTVlyP-VihU&t=403s):

<img src="./images/MAXOUT_DIAGRAM_MLFORNERDS.jpg" alt="Maxout Diagram" />

### 📝METHODOLOGY
Both models will be trained on Hadnwritten digit classification via MNIST (344k samples). 

The ReLU model will be trained via FastAI, whereas the Maxout model will use FastAI *Dataloaders* and *Learners* but PyTorch for the training loop (as the activation function has been implemented seperately). In the end, **both models are tested on the same 8 000 samples** and accuracy + confusion matricies are reported.

Feel free to clone this repo and try it out for yourself! Reach out to aditya.dewan124@gmail.com, @adidewan124 on Twitter, or via LinkedIn if you have any questions or ideas.

*Special thanks to Ian Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio for coming up with the original concept back in 2013! Check out their paper here https://arxiv.org/abs/1302.4389*

### 💡KEY LEARNINGS / ABOUT.

This project is one of many attempting to expand my horizons in deep learning and beyond. Let me know if there are any other papers you'd like to see me replicate!

That being said - this one project provided invaluable lessons in terms of machine learning techniques. Namely:

1. 🧠 **The intricacies of FastAI and PyTorch.** Prior to this project, I worked primarily in TensorFlow - thanks to this project, I learned how to implement custom activation functions and layer models in Pytorch and use a mixture of mid-level FastAI APIs (such as datablocks and dataloaders) to create new and better models.

2. ✖️ **The mathematical fundamentals behind machine learning.** For instance, what activation functions actually do behind the scenes, how they affect feature space and model predictions, and the role they (Maxout in particular) play in both reducing training times and modelling more complicated functions.

3. ⚖️The drastic importance of **weight initialization**. Initially, the weight intitalizations of the Maxout function in Maxout.py were set to **random**, a practice which ended up generating hundreds of NaN and infinite values during training. Exploring weight initialization and different pooling techniques, the project taught me the value of **uniform and normal initializations** as well as best practices for both (such as setting the bounds to +- 1/sqrt(inputs) when dealing with uniform distributions). All in all, this was critical in helping me gain a **first-principles understanding of weight transformations.**

4. ✅**Good model evaluation technicques** - accuracies, confusion matricies, and the importance of model validation (as well as how to compare the performance of two models in a statistically sound manner; such as using the same test set with identical samples, disabling augmentation, and more).

Looking forward to doing more complex and challenging projects in the future! Let me know your thoughts.