# Stock Price Prediction using LSTM and Alpaca API

A machine learning appraoch to optimising word guessing strategies for the clasic Hangman game.
---

## Overview

The code utilises the 250k words found in a text file within the repository and uses them to train a neural network model (with an LSTM layer). From there, the model is loaded onto a jupyter notebook file where it can interract with an API to make guesses. This is excluded from the code to avoid IP issues.

The key steps include:

1. **Data Retrieval**: Using the Alpaca API to fetch historical stock data.  
2. **Preprocessing**: Scaling data and creating sequences for training. 
3. **Modeling**: Training a neural network with an LSTM layer.
4. **Prediction**: Evaluating the model and generating hangman guesses.  

---

## Additional methods

To demonstrate my comfort with a range of machine learing methodologies, I have curated simplified versions in the addition_methods folder. These include:

- **Reinforcement Learning**: To provide an alternate method and test accuracy.
- **Bi-LSTM**: to further work on the time-series dependency of guesses.
- **Bayesian**: Utilises probabilities through bi & trigrams.

---

## Purpose

This project serves as a demonstration of my fluency with all major machine learning algorithms, and leverage them as a tool for problem solving.

---

## Notes

- The code is structured for clarity and educational purposes.
- It does not work without a suitable API.
- The model can be further optimized or extended for production-grade applications.
