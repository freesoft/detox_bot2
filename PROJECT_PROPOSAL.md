PLEASE read following article briefly so that you will get some idea what to say in the proposal.<br/> This seciton is not a part of the prpopsal and will be deleted later.
- https://us.battle.net/forums/en/overwatch/topic/20759196162
- https://boards.na.leagueoflegends.com/en/c/player-behavior-moderation/Q4nTlURg-why-toxic-players-continue-to-be-toxic-from-a-toxic-playeros-perspective
- https://us.battle.net/forums/en/overwatch/topic/20747204974
- https://www.theguardian.com/games/2018/aug/17/tackling-toxicity-abuse-in-online-video-games-overwatch-rainbow-seige
- https://www.destructoid.com/a-discourse-on-discord-examining-online-toxicity-501984.phtml
- http://jultika.oulu.fi/files/nbnfioulu-201706022379.pdf
- https://www.rockpapershotgun.com/2018/03/05/rainbow-six-siege-toxic-chat-bans/


# Project proposal - Detox Bot
## CS498 Cloud Computing Application, Spring 2019

```
Wonhee Jung (wonheej2@illinois.edu)
Kevin Mackie (kevindm2@illinois.edu)
Cindy Tseng (cindyst2@illinois.edu)
Conrad Harley (harley3@illinois.edu)
```
# Overview

Since online services has been widely developed and being used, the toxicity or salty message in the chat, message, or comment
became a problem. 
Many online systems has its own profanity filtering embedded in their system, but those are all pre-registered word based
so that the filtering doesn't work for new words or expressions, which might not have any bad word in it at all, 
but it's toxic.

Also, every system should have developed their own profaility filtering system, although it's functionality is not much different than others and actually good candidate to be PaaS or SaaS service.

[TODO] You probably get some idea what I'm trying to say. Please change the sentence that sounds good and cool, but stay focus on minimizing the work we need to for final project... Check https://www.coursera.org/learn/cs-498/supplement/ewbEI/course-project-overview for what project proposal want us to do. 

# What we are going to make 

That said, our team is going to make a prototype of PaaS/SaaS service that provides

* machine leanring based toxicity/salty chat filtering engine
* and integerated application that uses the engine ( probably simple web chat app and things like chatbot )

# dataset

Unfortunately, there are very limited toxic/comment we can utilize. We weren't able to find those dataset exists in public.<br/>
Only thing we found so far is Kaggle's toxic comment classification challenge dataset and we are going to use the set to train our classifier. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# Solutions for the project

The team ill use following technologies and solutions to be able to cover high volume/traffic in the future.

* Docker - to distribute the engine/app easily  to mulitple instance 
* Kubernetes - make deploy easy and quickly scale out
* CI/CD pipeline - automated build
* AWS, GCP, Heroku, etc - depends on what/how we want to make an app and run, we are going to use one or two IaaS/PaaS services.
* HDFS( or compatible ) - to store big data and share it between systems
* Apache Spark (and MLlib ) - planning to use Scikit-learn for machine learning part, but in case it's not enough to handle 
huge volume and traffic, then we will consider to use Apache Spark + MLlib to imiplement the engine
* REST APIs - Depends on how we want to provide a functionality of the filteirng engine to other applicaiton, we might need to
provide HTTP Rest APIs or something similar RPC API.


