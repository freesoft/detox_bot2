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

Online platforms allow people to express their opinions freely, and stimulate collaboration across the globe. 
Unfortunately, online interaction may often come with loosened inhibitions in making profane, bigoted, or offensive 
remarks. We refer to such unwelcome remarks as "toxic chat". 

Online systems may or may not have their own embedded profanity filtering, and those that do typically use
pre-registered terms and simple pattern matching. This approach lacks the deeper contextual understanding needed to 
identify sentences that are toxic but that may not contain banned terms. 

Thus we propose a new toxic chat filtering system that differentiates itself in that a) its filtering is based on
machine learning and deeper contextual analysis, and b) it is deployed as a scalable and easily integrated web
framework that can be adapted to any source of text for online interaction of any size. 

The platform will be based on Docker and Kubernetes for easy portability between different types of systems. It will use 
state-of-the-art distributed systems technology for processing and storage (HDFS), to allow for rapid scaling to any 
size while maintaining a shared file space (HDFS) between each Kubernetes Zone.

The framework will be documented in a final report that presents the architecture, development, and use of this system
in the context of a web chat application and Twitch chatBot as motivating examples.

[TODO] You probably get some idea what I'm trying to say. Please change the sentence that sounds good and cool, but stay focus on minimizing the work we need to for final project... Check https://www.coursera.org/learn/cs-498/supplement/ewbEI/course-project-overview for what project proposal want us to do. 

# What we are going to make 

That said, our team is going to make a prototype of PaaS/SaaS service that provides

* machine leanring based toxicity/salty chat filtering engine
* and integerated application that uses the engine ( probably simple web chat app and things like chatbot )

# dataset

Unfortunately, there are very limited toxic/comment we can utilize. We weren't able to find those dataset exists in public.<br/>
Only thing we found so far is Kaggle's toxic comment classification challenge dataset and we are going to use the set to train our classifier. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

# Solutions for the project

The team will use following technologies and solutions to be able to cover high volume/traffic in the future.

* Docker - to distribute the engine/app easily  to mulitple instance 
* Kubernetes - make deploy easy and quickly scale out
* CI/CD pipeline - automated build
* AWS, GCP, Heroku, etc - depends on what/how we want to make an app and run, we are going to use one or two IaaS/PaaS services.
* HDFS( or compatible ) - to store big data and share it between systems
* Apache Spark (and MLlib ) - planning to use Scikit-learn for machine learning part, but in case it's not enough to handle 
huge volume and traffic, then we will consider to use Apache Spark + MLlib to imiplement the engine
* REST APIs - Depends on how we want to provide a functionality of the filteirng engine to other applicaiton, we might need to
provide HTTP Rest APIs or something similar RPC API.


