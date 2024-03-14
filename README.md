# Hate-Speech-Classification

The freedom of speech is a well guarded right across many countries in the world, but certain boundaries always need to be set. The rise of plenty of social media apps and online communities has given people a medium to express their opinions and interact with each other on any number of topics. Some discussions are polite and gentle while others turn ugly. People turn provocative against each other over petty disapprovals and start hurling abuses and hateful comments. Such type of negativity has a bad impact on online commuity and may lead to bad consequences. 

Such community based apps often issue a code and conduct manual going against which results in a permanent account ban. They also give other users the option to report a complaint against a toxic user. The companies go through the comments and posts of that user to verify people's complaint. Since such big companies have to deal with many complaints in a given day, there is no possible way that they can manually verify each and every account. Thus the companies leverage the use of **sentiment analysis**. 

Sentiment analysis is a Natural Language Processing (NLP) technique. It is a supervised learning algorithm that is trained on a text and its label. Basically, It classifies the text into two or more than two categories depending on the labels. Classifying hate speeches is a binary classification problem, a speech can either be hateful or not hateful. 

Just to replicate how a standard hate-speech classifier works, I have built an end to end application demonstrating it.

For this task, I obtained the Twitter's hate-speech classification data from Kaggle and uploaded it on GCP Bucket for retrieveing it as and when required. Though it contains several features, but we are only concerned with two, the text with people's tweet and their labels as hateful/non-hateful. We get an unprocessed dataset having missing and duplicate values. We first deal with them ,remove the unwanted features and ensure that the dataset is clean with proper binary labels assigned. You can check out analysis notebook in the research directory. 

Next we clean the text data. Text-cleansing process in NLP usually follows a certain set of protocols. We utilise a python library called **NLTK** for this task. It has all the built-in text cleaning modules available. The standard cleaning procedure follows:

* Basic text-preprocessing
  
      * stop-word removal i.e. removing very common words, usually conjunctions like 'the','and','or'...
  
      * stemming/lemmatization for converting the words into their root form (Ex. running -> run)
  
      * punctuation and symbol removal (:,;,.,#,@...)
  
      * lower casing (converting the words into lower case) 

* Advance text-preprocessing
  
      * Part-of-Speech tagging i.e. assigining the grammatical categories to words to maintain the context of sentence
  
      * Parsing which is again the process of breaking down a text into related words and identifying the sentence's grammatical structure.
  
      * Correference resolution. Consider a sentence "Charlie is a dog and Doug is his owner." Here Charlie is a male dog as we can understand from the context.

Once we are done with the cleaning, we move onto feature engineering. Any Machine Learning model wont understand a text directly and only understands it as a vector. This step is called tokenization where we convert a text into vector. There are different ways to perform text vectorization: bag-of-words (BoW), TFIDF (Term-Frequency-Inverse-Document-Frequency), One-hot Encoding, word2vec, and embeddings. I chose to go with embedding vectorization. 

**Embeddings** refer to the dense vector representations of words, phrases, or documents learned from a large corpus using techniques like Word2Vec, GloVe, or FastText. Unlike BoW or One-Hot Encoding, embeddings capture both syntactic and semantic relationships between words. Embeddings are typically trained using neural network-based models on large text corpora, and the learned representations are used as input features for downstream tasks. They are effective in capturing the meaning of words, handling out-of-vocabulary words, and improving the performance of machine learning models in various natural language processing tasks.

![keras_embeddings](https://github.com/shazam37/Hate-Speech-Classification/assets/119686545/f70775de-b788-44b2-af69-1e9617b3fe72)

Tensorflow's keras library provides an embedding module where we need to specify the length of vector we desire for storing the text info. More the amount of text, more should be the vector-size so that it can encode all the words. 

We encode the binary labels with one hot encoding. After performing all the given steps, our data is now read to be trained on a deep learning model. 

Recurrent Neural Networks (RNN) are used for training on a sequential information. It is the basic building block behind all the state of the art NLP models. RNN just like any neural network takes in a vectorized text input and predicts the next text output. But the special thing is that the output is then fed again along with the next input to the model, that way, the model is also trained on the previous information. This method of processing information accounts for the order of sequence which is the whole crux of any text information. 

![rnn](https://github.com/shazam37/Hate-Speech-Classification/assets/119686545/4c40ebbd-0527-44bf-a89a-e6d9ed260015)

RNNs can even take information in different ways as can be seen from the image below:

![tensorflow-types-of-rnn](https://github.com/shazam37/Hate-Speech-Classification/assets/119686545/21d7980d-dc98-4668-aee5-a2ebe6ac9b76)

The downside of a simple RNN is that it doesn't take into account the long term dependencies of a sequence. It is so dynamic that it forgets the old information in favour of a new one. To solve this problem, people have tweaked into RNN and have come up with different algorithms to keep track of long term dependencies. The most famous ones being: GRU, LSTM, and Transformers. For my task, I chose to go with LSTM because of the not so big size of my text data. An LSTM is enough to keep track of dependencies in my case. 

The LSTM architecture looks like a bunch of pipes going here and there each having a different functionality, and that's what precisely an LSTM is. 

![lstm](https://github.com/shazam37/Hate-Speech-Classification/assets/119686545/226562b9-3cb9-4a90-bfc9-abd4e29734d6)

Consider the name of the different components in a literal sense. Input pipe/state takes in the input while the memory pipe/state carries the past information. The ones in between are the gates that decides what will happen to the information. All the components can be summarised as:

* Input Gate (i):

The input gate determines how much of the new input data should be let through to update the memory state. It takes the current input and the previous hidden state as input and produces a value between 0 and 1 for each number in the memory state. A value of 0 means "let nothing through", while a value of 1 means "let everything through".

* Forget Gate (f):

The forget gate determines what information from the memory state should be thrown away or kept. It takes the current input and the previous hidden state as input and produces a value between 0 and 1 for each number in the memory state. A value of 0 means "completely forget this", while a value of 1 means "keep this completely".

* Memory State (C):

The memory state is like a conveyor belt that runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged. The input gate and the forget gate help in deciding how to update the memory state.

* Output Gate (o):

The output gate determines what parts of the memory state should be output as the hidden state. It takes the current input and the previous hidden state as input and produces a value between 0 and 1 for each number in the memory state. The hidden state, which is also the output of the LSTM unit, is a filtered version of the memory state.

* Hidden State (h):

The hidden state is the output of the LSTM unit. It's a filtered version of the memory state that focuses on the parts that the output gate decided to output.

It's a little tricky to get at first but we can appreciate the magnificence of this algorithm. It has a memory!! 

Just like any neural network architecture, we specify the number of LSTM layers we want, the amount of dropout regularization we want (to prevent overfitting), etc. We finally add a Dense layer at the end with a sigmoid activation to generate a binary output.  

For my case, I chose to go with 1 layer and a droput rate of 0.2. I then compiled the model with a binary-crossentropy loss, RMSProp optimizer, and accuracy as an evaluation metrics. With jsut 5 training epochs with a batch size of 128 and a validation split of 0.2, I obtained a validation accuracy value of 96%. 

The trained model was wrapped up as an application using FastAPI and is ready to be deployed. The application follows proper MLOPs principle with CI/CD deployment being carried out with CricleCI. You can install the required dependencies and run the app.py for launching the application. 

The app interface allows you to train the model yourself (You can tune the hyperparameters in the config file):

![Screenshot from 2024-03-14 16-22-36](https://github.com/shazam37/Hate-Speech-Classification/assets/119686545/5e991858-710c-4f11-aee0-f3a6fd6cc94b)

And there is a prediciton interface where we need to specify the text for prediction. Let's say we give a text like:
"rt urkindofbrand dawg rt you ever fuck a bitch and she start to cri you be confus as shit". Its clearly a abusive statement. See what the model has to say about it:

![Screenshot from 2024-03-14 16-26-41](https://github.com/shazam37/Hate-Speech-Classification/assets/119686545/ede9864d-62ac-4b45-a7aa-dfbc3efce183)

Voila! It classfied correctly! We now have a speech buster in the house B-)

The given template can be extended for classifying any text into any category given we have a properly labelled data for it. Now you know the power of such NLP algorithms running day and night to make this world a better place. Happy classifying!












