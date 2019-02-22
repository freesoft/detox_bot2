## Team 15, Project proposal - Detox Bot
### CS498 Cloud Computing Application, Spring 2019

```
Wonhee Jung (wonheej2@illinois.edu), Kevin Mackie (kevindm2@illinois.edu), 
Cindy Tseng (cindyst2@illinois.edu), Conrad Harley (harley3@illinois.edu)
```
## Overview

Online platforms allow people to express their opinions freely, and stimulate collaboration across the globe. 
Unfortunately, online interaction may often come with loosened inhibitions in making profane, bigoted, or offensive 
remarks. We refer to such unwelcome remarks as "toxic chat". 

Online systems may or may not have their own embedded profanity filtering, and those that do typically use
pre-registered terms and simple pattern matching. This approach lacks the deeper contextual understanding needed to 
identify sentences that are toxic but that may not contain banned terms. 

Thus we propose a new toxic chat filtering system that differentiates itself in that a) its filtering is based on
machine learning and deeper contextual analysis, and b) it is deployed as a scalable and easily integrated web
framework that can be adapted to any source of text for online interaction of any size. 

The platform will be based on Docker and Kubernetes for easy deployment and dependency management and to allow for fast scale-out
to large systems. It will use 
state-of-the-art distributed systems technology for processing and storage, to allow for rapid scaling to any 
size while maintaining a shared file space (HDFS) between each Kubernetes Zone.

The framework will be documented in a final report that presents the architecture, development, and use of this system
in the context of a web chat application and Twitch chatBot as motivating examples.

## What we are going to make 

We will create a prototype of PaaS/SaaS service that provides the following specific capabilities:

* machine-learning-based toxic chat identification and filtering engine
* integerated web chat application or chatbot that uses the engine to analyze a real-time stream of text

## Dataset

The only publicly available toxic comment dataset we have been able to find so far is Kaggle's toxic comment classification challenge dataset, which we will use to train our classifier. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## Cloud system integration and development

The team will integrate the following technologies and solutions to provide a general framework that can scale to high volume/traffic in the future.

* Docker, Kubernetes - easy deploy and scale out
* CI/CD pipeline - to automate the build, integration, and deployment
* AWS, GCP, Heroku, etc - to deploy the solution into a mainstream PaaS infrastructure 
* HDFS or similar - to store big data and share it between systems
* Scikit-learn or Apache Spark + MLlib - machine learing for the detox engine
* RESTful APIs - to help other applications integrate detox engine in the system
