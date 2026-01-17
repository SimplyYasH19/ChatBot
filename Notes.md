Project :  ChatBot





This chatbot is not ChatGPT-style generative AI. It is: A supervised machine learning classifier that:



* Takes a user sentence
* Converts it into numbers
* Predicts which intent the sentence belongs to
* Prints a pre-written response for that intent





So the bot does not think. It only:  Classifies → Looks up → Responds





Elementary” = rule + ML based, not LLM.





\## Technologies Used (Why Each Exists)



* Python  →   Core language
* NLTK    →    Text processing (tokenize, stem, normalize)
* NumPy   →    Numeric arrays
* Scikit-learn  →   Utilities (not core here)
* TensorFlow / Keras   →   Neural network classifier
* Colorama   →    Colored terminal output (cosmetic)







\##  logical phases:



* Install \& import libraries
* Define training data (intents JSON)
* Text preprocessing (tokenization, stemming)
* Feature engineering (Bag of Words)
* Train neural network
* Chat loop (prediction + response)





“This project implements an intent-based NLP chatbot using supervised machine learning. It uses NLTK for preprocessing, Bag-of-Words for feature extraction, and a feedforward neural network for intent classification. The predicted intent is mapped to predefined responses stored in a JSON structure.”





“This chatbot is a retrieval-based, intent-driven system. It uses a classifier to map user input to an intent, and then retrieves a predefined response associated with that intent.”











##### \## **Step by Step Explanation** :





1. !pip install tensorflow>=2.12 nltk>=3.8 colorama>=0.4.4 numpy>=1.22 scikit-learn>=1.2 Flask>=2.3



 	. These libraries are REQUIRED for this project to run

 	. The >= means:

 		“Any version equal to or newer than this is acceptable.”



 	a) TensorFlow:  it acts as a massive library of ready-made mathematical "blocks" that you can wire together to build an AI, rather than writing the complex math from scratch.

 

 		      . Preprocessing Data: It takes raw information and turns it into Tensors—basically just organized grids of numbers that the computer can understand.

 		      . Building the "Brain": Using a sub-tool called Keras, you stack "layers" together to form a neural network.

 		      . Training: This is the "heavy lifting." You show the model thousands of examples (e.g., "this is a cat photo").

 			TensorFlow does the complex calculus behind the scenes to help the model learn patterns and correct its own mistakes.



 	b) NLTK: \[ Natural Language ToolKit ] :  open-source library for the Python programming language



 		  . its a text processing toolkit .

 		  . it splits the sentence in words

 		  . Converts those words to root form

 		  . Clean the text \& normalise the input



 	c)  NumPy:  library for Python used to store, process, and compute large amounts of data efficiently



 		  . Numerical data handling

 		  . Stores data in fast, compact arrays (much faster than normal Python lists)

 		  . Performs massive mathematical operations on that data (matrix math, statistics, algebra, transformations)

 		  . Store vectors (Bag of Words)

 		  . Stores training data



 	d)  Scikit Learn:

 			  . machine learning library for Python that provides ready-made algorithms for prediction, classification, clustering, and data analysis.

 			  .  its not the main core library , its just a support

 			  .  The main one is  = Tensorflow



 	e) Colorama:

 			. A utility library for coloring and formatting terminal text in Python.

                        . Because plain terminal output is hard to read and debug.

 			. Highlight errors in red . Show success in green

 			. Show user vs bot messages in different colors



 	f) Flask = Deployment																					. To provide a simple, minimal way to expose Python code to the web without heavy complexity.

 			. server that runs the chatbot

 			. used to build REST API's,  Deploy ML models,   Create dashboards and backend services



 







2]  nltk.download('punkt')  :  A pre-trained sentence \& word tokenizer model.

 

 			. when we write : nltk.word\_tokenize(sentence) , this function Uses punkt internally to split text into words correctly.

 

    nltk.download('wordnet')  :  database of English words (dictionary + relations)



 			. When we write : WordNetLemmatizer()   or   LancasterStemmer() / WordNetLemmatizer

 			. it helps to reduce words into short like:

 								cars → car

 								running → run



   nltk.download('omw-1.4')  :  . Open Multilingual WordNet data

 				. WordNet depends on this dataset internally

 				. provides Word meanings across languages



 	\*\* NLTK requires external linguistic datasets such as punkt for tokenization and WordNet for lemmatization. These resources are downloaded using nltk.download() 		before performing text preprocessing.”







3]  JSON:  Java Script Object Notation  .  It is a text file format used to store and exchange structured data.



 	Why we need it : Your chatbot needs:

 						A persistent knowledge base

 						That stores:

 							Intents

 							Patterns

 							Responses













##### \## **Next Part/Cell**



 			“This cell loads the intent data, extracts patterns and labels, encodes the labels numerically, tokenizes the text, and

 			 converts all sentences into fixed-length numeric sequences suitable for neural network training.”







 											Meaning:

1.  import numpy as np                                                                         numpy          →  for numeric arrays
   from tensorflow.keras.preprocessing.text import Tokenizer                                  Tokenizer      →  converts text → numbers
   from tensorflow.keras.preprocessing.sequence import pad\_sequences                          pad\_sequences  →  makes all sentences same length
   from sklearn.preprocessing import LabelEncoder                                             LabelEncoder   →  converts class labels (strings) → numbers



2\.  with open('intents.json') as file:                                       Meaning:   Open the JSON file and load it into Python dictionary data.

    data = json.load(file)





3\.  Create Empty Containers:                    training\_sentences = \[]                   training\_sentences =  all patterns (inputs)

 						training\_labels = \[]			  training\_labels    =  their tags (outputs)

 						labels = \[]				  labels             =  unique list of tags

 						responses = \[]				  responses 	     =  responses for each tag





4\. Extract data from JSON:

 			     for intent in data\['intents']:				  For each intent:

    				for pattern in intent\['patterns']:				Take each example sentence → add to X

        			   training\_sentences.append(pattern)				Take its tag → add to Y

                                   training\_labels.append(intent\['tag'])			Store responses

    				responses.append(intent\['responses'])				Store unique tag list

    				if intent\['tag'] not in labels:

        			   labels.append(intent\['tag'])



5\. Convert labels to numbers:

 				lbl\_encoder = LabelEncoder()						Meaning:

 				lbl\_encoder.fit(training\_labels)						"greeting" → 0

 				training\_labels\_encoded = lbl\_encoder.transform(training\_labels)		"goodbye" → 1      "thanks" → 2







6\.  Tokenization Setup:      vocab\_size = 1000  : Limits the tokenizer to the 1000 most frequent words in the dataset.

 						  This directly controls memory usage, model size, and how much of the language the model can represent.

 			     embedding\_dim = 16  :  Defines how many numbers are used to represent each word internally.

 					            Larger values allow richer semantics but increase model size and training cost.



 			     max\_len = 20         :  Forces every input sentence to be exactly 20 tokens via padding or truncation.

 						     This sets a hard limit on how much context the model can see.



 			     oov\_token = "<OOV>"    :   Any word not in the known vocabulary is replaced with this token.

 							This prevents failures and caps vocabulary growth but collapses all unknown words into one bucket.





7\.   Create Tokenizer :     tokenizer = Tokenizer(num\_words=vocab\_size, oov\_token=oov\_token)                Means : Read all sentences and build a word dictionary.

 			    tokenizer.fit\_on\_texts(training\_sentences)





8\.  Convert Text to Number :  sequences = tokenizer.texts\_to\_sequences(training\_sentences)





9\.  Padding - helps to generate same length inputs, as neural netw. requires it  : padded\_sequences = pad\_sequences(sequences, truncating='post', maxlen=max\_len)









##### **## Next Cell**



			**A sequential neural network is built using an embedding layer, pooling layer, and dense layers to classify input sentences** 

			**into intent categories.   The model is trained using categorical crossentropy and Adam optimizer for multiple epochs.”**





&nbsp;	1. from tensorflow.keras.models import Sequential						Sequential =  A simple stack of layers (one after another)

&nbsp;	   from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D			Dense      =  Fully connected neural network layer

&nbsp;													Embedding  =  Converts word IDs into meaningful vectors

&nbsp;												GlobalAveragePooling1D  =  Compresses sequence into a single vector





##### \## Next Cell  



&nbsp;				This cell serializes and saves the trained neural network model and preprocessing objects so they can be 

&nbsp;				reused for inference without retraining.







 	1. Import pickel  :    pickel = Its a python tool,

&nbsp;			       Converts Python objects (in memory) → into files on disk



&nbsp;	2. with open('tokenizer.pickle', 'wb') as handle:

&nbsp;   		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST\_PROTOCOL)



&nbsp;				. Save the Tokenizer obj. to a file , we basically freeze it 



&nbsp;				. Why?, Used as -   same word → number mapping MUST be used during chat time.



&nbsp;				. If you regenerate tokenizer: Number changes \& Model Breaks



&nbsp;	3.  with open('label\_encoder.pickle', 'wb') as enc\_file:

&nbsp;		 pickle.dump(lbl\_encoder, enc\_file, protocol=pickle.HIGHEST\_PROTOCOL)



&nbsp;				. Means  =  Save the label encoder (intent → number mapping)



&nbsp;				. Why?, As: Model outputs numbers, not intent names.















Browser (HTML/CSS)

&nbsp;       ↓

&nbsp;     Flask (Python)

&nbsp;       ↓

&nbsp; Your ML Model (keras)

&nbsp;       ↓

&nbsp;     Response

&nbsp;       ↑

&nbsp;     Back to UI

&nbsp;



















