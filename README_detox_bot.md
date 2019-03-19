# "Detox" toxic chat classifier

## Overview

This project is for my University of Illinois at Urbana-Champaign MCS-DS Fall 2018 CS410 project submission. It is required by UIUC that the project need to be uploaded on public Git repository. After getting grade by UIUC, I'm going to keep working on improving the code.

The project has following components :

* Detox core engine : TF-IDF + Multinomial Naive Bayes based trained classifier.
* Web chat : Python Flask-socketIO based functioning multi web chat that has Detox integrated. 
* TwitchTV chatbot : Twitch TV Chatbot that has integrated with Detox. It's able to connect to the channel, listen the message, and decide if chat is toxic.

Both web chat and Twitch TV chatbot are implemented for Detox testing. ( Don't expect any fancy design or UX though! ) 

Precision/Recall/F1-Score : Shuffled the training set and took 20% of it as comparsion using scikit-learn's classification_report, and here is the result. The data can vary slightly every time and also depends on what kind of dataset is used for training.


|     | precision | recall | f1-score | support |
-----|-----------|--------|----------|----------
   0 |  0.94     | 1.00   |  0.97    | 28629   
   1 |  0.95     | 0.48   |  0.64    | 3286    
avg/total|  0.94     | 0.94   |  0.94    | 31915  


### Initialization process

![diagram 1](/diagram/overview_diagram.png)

A few customizations are done here.<br/>

* stopwords.txt modification : original stopwords.txt was from CS410 class' PyMetaPy. However, it turned out the target documents that the project was aiming, which is online chat/in-game chat/etc, include so many slang words that doesn't add much meaning in it or simply abbreviation-ish. Add dozens of those "useless" slangs in the file so that tokenizer and vectorizer can focus more on their work.

### How the core engine is integrated


![diagram 2](/diagram/overview_diagram_2.png)

## Features

* The Detox uses TF-IDF, that enables it to penalize and rewards those frequent/rare words along with frequence of words in docs/message.
* Uses Multinomial Naive Bayes classifier, that enables Out-of-core model fitting, a.k.a incremental training when new training data is available.
* Minimum memory footprint while training is available. (It has some drawback so it's turned off now, turning on requires code change at this point. ) 
* Reuse trained classifer engine and also fitted vectorizer so it runs faster after initial execution.
* Integerated Twitch Chatbot so Moira can be deployed to any TwitchTV channel and determine if chat is toxic.
* Integrated simple web chat app and TwitchTV chatbot.
* ***Python 3*** only.


## Quick startup & test by web chat 

The Detox and its webchat app is deployed on Heroku. [https://uiuc-cs410-detox.herokuapp.com](https://uiuc-cs410-detox.herokuapp.com) <br/>
Just open the url from your web browser, and try chat. It's ugly webchat but actually functioning for multi chat. Try open a few more tab on your browser and see how toxic classifier works. For any toxic chat that Detox recoginze, the messsage will be displayed in red with "toxic" prefix. **Recommended** if you only need to test how the classifier works.

![diagram 2](/diagram/overview_chat.png)


## Other ways to install & test ( won't be happy about this! )

The project has Dockerfile script that you can use for Docker image build. Also, I've deployed pre-built docker image on public Docker Hub repository.

### 1. build Docker image and start

Install [Docker](https://www.docker.com) for your macine and checkout the git repo. Once it's ready, then run

```
docker build -t uiuc-cs410-fall2018:detox .
```
Yea, that's "." at the very end and one space between "x" and ".". So careful not to omit ".". The build will take some time.<br/>
Once the build is done, you'll be able to see some hash value at the end of the build. ( or `docker images` and get the image id with the latest build ).<br/>
With those image id, run

```
docker run -it -d <image_id>
```
Startup will take some time. If you want to see what's happening during the docker run, omit `-d`. <br/>
If startup success, then access to `localhost:5000` on your web browser. If web chat UI doesn't show up, then there is something wrong but I won't try to explain what could go wrong here since it will take too much time. Just use Heroku instance for testing!!!

One possible problem is, the Detox requires lots of memory during the training and it **could*** kill the docker build OR docker container start up fail. Try to replace the training data file with smaller one if you really want to go with this way.

### 2. Use pre-built Docker image

Just in case you have a problem with building the docker image in your local, you can download pre-built image from [here](https://hub.docker.com/r/freesoft/uiuc-cs410-fall2018).<br/>
To pull the image, run

```
docker pull freesoft/uiuc-cs410-fall2018
```
Once downloaded, runing step is same as 2.


### 3. Just run the project as it is

After git clone this project, simply run

```
python3 ./webapp.py
```

to train the model before running the test app. The file has a few dependencies so you probably need to install it. Required python libraries are specified in `requirements.txt`, but you will need to install `pip3` and some others as well.


## Twitch TV Chatbot

The Detox comes with tiny TwitchTV chatbot(`chatbot.py`), so you can use it to collect livefeed the chat logs into the Detox. This doesn't have any web UI integratio done so far, so it needs to be run locall console. 


![diagram 3](/diagram/overview_twitch_chatbot.png)

Either docker attach to the running contianer or in your local, type

```
python chatbot.py <username> <client id> <oauth2 acccess token> <channel name>
```
and you'll see that the program is collecting the live chat log from the TwhtchTV channel, and determine if chat is toxic or not.<br/>
It will also create a file with same channel name that Chatbox connected(`<channel name>`) and start logging the message.

Parameter explanation:

* `<username>` : Your username on Twitch. 
* `<client id>` : visit [https://glass.twitch.tv/](https://glass.twitch.tv/) and login with your own Twitch account. Once you create new app, you'll be able to get Client ID on Dashboard -> App section. <br/>
* `<oauth2 access token>` : visit [here](https://twitchapps.com/tmi/#access_token=flwh72scl6503e6bs2xnwl6g6l5jeu&scope=chat%3Aread+chat%3Aedit+channel%3Amoderate+chat_login&token_type=bearer) and click "Connect with Twitch", and use it for `<oauth2 access token>`.
* `<channel name>` : TwitchTV channel name you'd like to deploy the Detox engine. Use the channel name you can check from web browser's url, which is generally all lowercase regardless of what you can see on twitch user's dashboard.



## FAQ

### I'd like to train the classifier with my own data. What should I do?

If you want to see how it generates classfier and train it, not using pre-existing one, simply delete `classifier.joblib` and `vectorizer.joblib` and re-run the above shell command.

Trainig data and test data files are stored in `data/` directory, so if you wnat to test the engine with your own file, simply replace the file with your own. Make sure you match the CSV file format. <br/>
First line should start with following header title,

```
"id","comment_text" 
```
and from 2nd line you can use your own data. The engine doesn't care much about the value in the "id" part and add whatever chat logs you'd like to test.

Training data file has similar format, but it has additional CSV column that needs to be pre-labeled by human so that it can be used for traning purpose.

### I'd like to know more about Twitch TV Chatbot. Where I can find the relevant docs?

Check out [https://dev.twitch.tv/docs/irc/guide/](https://dev.twitch.tv/docs/irc/guide/). You can find more on their [dev site](https://dev.twitch.tv/docs/.

### Where I can contact for aksing questions or issue to report?

Please [open a new issue](https://github.com/freesoft/detox_bot/issues/new) on this Github repository and I'll take a look shortly.
 
## Resources ##

* NLTK : https://www.nltk.org
* Scikit-learn : http://scikit-learn.org
* Docker : http://docker.com
* Heroku : http://heroku.com
* Docker Hub : http://dockerhub.com

