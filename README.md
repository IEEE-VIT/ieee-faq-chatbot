<p align="center"><img width="40%" src="https://hacktoberfest.digitalocean.com/_nuxt/img/logo-hacktoberfest-full.f42e3b1.svg"/></p>

<p align="center">Answer all frequently asked questions!</p>

# IEEE FAQ Chatbot
***
## About 
* This is a Frequently Asked Questions(FAQ) answering chatbot, customly built to answer any question related to IEEE-VIT. 
* This project is built from scratch without using pre existing platforms such as [Dialogflow](https://cloud.google.com/dialogflow/docs) or [Amazon Lex](https://aws.amazon.com/lex/).
* Further, an API is created for the same using [FastAPI](https://fastapi.tiangolo.com/).
## Model
* Two separate models were built for intent classification and are available in [model_training directory](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training):
 1. [TensorFlow Model](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training/tensorflow_model) 
 2. [PyTorch Model](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training/pytorch_model)
* Both the models are independently working and perform the same action of classifying input query into fixed set intents.
## Data
* The [Data](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training/data) used to train both the [TensorFlow Model](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training/tensorflow_model) and [PyTorch Model](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training/pytorch_model) was same and customly generated as per requirements. 
* The Data had to be augmented before training the models as [manually generated data](https://github.com/IEEE-VIT/Chatbot/blob/master/model_training/data/raw_data.csv) was insufficient. 
* The [Data Directory](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training/data) inside [model_training directory](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training) has [notebook for augmenting](https://github.com/IEEE-VIT/Chatbot/blob/master/model_training/data/augmenting_data.ipynb) and [augmented data](https://github.com/IEEE-VIT/Chatbot/blob/master/model_training/data/augmented_data.csv) as well.
## Installation and Usage
Before proceeding make sure you have [Python 3.8 or above](https://www.python.org/downloads/) installed. 
  ### Install and use created API
  1. Clone this repository:
  ``` 
  git clone https://github.com/IEEE-VIT/IEEE-FAQ-Chatbot.git
  ```
  2. Create a python virtual environment and activate it:
  ``` 
  pip install virtualenv 
  ```
  ```
  python -m venv myenv 
  ```
  ```
  myenv\Scripts\activate 
  ```
  3. Install the requirements:
  ```
  pip install -r requirements.txt
  ```
  *Important note: for installing torch 1.7.1, direct wheel file was used in [requirements.txt](https://github.com/IEEE-VIT/Chatbot/blob/master/requirements.txt) which is python version and OS specific. The used ```.whl``` file is for Python 3.8 and Linux OS. If you have a different version of Python or OS installed, you need to replace the existing ```.whl``` file with the correct one in [requirements.txt](https://github.com/IEEE-VIT/Chatbot/blob/master/requirements.txt). All other files for different Python versions and OS can be found [here](https://download.pytorch.org/whl/torch_stable.html). Make sure to get correct torch version (1.7.1) as well for PyTorch model.

  4. Start the server on localhost:
  ```
  uvicorn main:app --reload
  ```
  Once the server has started successfully, go to http://127.0.0.1:8000/docs to test your API. 8000 is the default port, which can be changed in [main.py](https://github.com/IEEE-VIT/Chatbot/blob/master/main.py) to any unused port.

## Install using docker
  To install via docker, make sure you have [docker desktop](https://www.docker.com/products/docker-desktop) installed if you are working on windows.
  1. Clone this repository:
  ``` 
  git clone https://github.com/IEEE-VIT/IEEE-FAQ-Chatbot.git
  ```
  2. cd into root directory and build the image using Dockerfile:
  ```
  docker build -t myimage .
  ```
  3. Once the image has been built, start the container:
  ```
  docker run -d --name mycontainer -p 8000:8000 myimage
  ```
  This will start the container at localhost port 8000. Go to http://127.0.0.1:8000/docs to test your API. 8000 is the default port, which can be changed in [Dockerfile](https://github.com/IEEE-VIT/Chatbot/blob/master/Dockerfile) to any unused port.

## Contributing
1. Fork this repository.
2. Clone the forked repository in your local machine and install it using above mentioned steps.
3. Create and checkout a new branch using:
``` 
git branch -m new-feature
```
You can use any branch name you like!

4. make all the changes you think will help improve this project! Now, open a PR by following these commands:
```
git add .
```
```
git commit -m "a short description about your commit"
```
```
git push -u origin new-feature
```
5. Now, go to your browser and open the forked repo and then raise a PR to ```master``` branch of this repo.

That's all. Now just hang tight while our maintainers review your PR and merge and close them! 

If you are new to contributing, checkout [contributing guidelines](). Do checkout issues labelled as ```hacktoberfest``` for some goodies and a T-shirt!
***
*Note: Training the models is not required for installing and using the API with above mentioned steps. This is because, both [tensorflow_model.py](https://github.com/IEEE-VIT/Chatbot/blob/master/tensorflow_src/tensorflow_model.py) and [pytorch_model.py](https://github.com/IEEE-VIT/IEEE-FAQ-Chatbot/blob/master/pytorch_src/pytorch_model.py) directly load the trained and saved model with ```.h5``` and ```.pt``` extensions respectively. 
  
  If you wish to see the training code, it can be found inside [model_training directory](https://github.com/IEEE-VIT/Chatbot/tree/master/model_training).
***
<p align="center">Consider leaving a :star: if you liked the project and organization :)</p>