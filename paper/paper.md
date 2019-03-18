---
journal: true
title: A Distributed Architecture for Toxic Chat Detection
author:
- name: Wonhee Jung (wonheej2@illinois.edu) 
  membership: University of Illinois Urbana-Champaign
  affiliation: University of Illinois Urbana-Champaign
- name: Kevin Mackie (kevindm2@illinois.edu)
  membership: University of Illinois Urbana-Champaign
  affiliation: University of Illinois Urbana-Champaign
- name: Cindy Tseng (cindyst2@illinois.edu)
  membership: University of Illinois Urbana-Champaign
  affiliation: University of Illinois Urbana-Champaign
  secondlast: true
- name: Conrad Harley (harley3@illinois.edu)
  membership: University of Illinois Urbana-Champaign
  affiliation: University of Illinois Urbana-Champaign
  last: true
date: May 1st, 2018
abstract: This paper presents the architecture, development, and use of a distributed system for toxic chat filtering. This system differentiates itself in that a) its filtering is based on machine learning and deeper contextual analysis, and b) it is deployed as a scalable and easily integrated web framework that can be adapted to any source of text for online interaction of any size. The platform is based on Docker and Kubernetes for easy deployment and dependency management, and uses state-of-the-art distributed systems technology to allow for fast scale-out to large systems. The system is presented in the context of a web chat application and Twitch chat bot as motivating examples.
IEEEkeywords: toxic comment, web, chat, detoxifier, detection, classifier, distributed, architecture
package:
- name: hyperref
  options: pdftex,bookmarks=false
- name: fontenc
  options: T1
- name: inputenc
  options: utf8
---

Introduction
============

\IEEEPARstart{O}{nline} <!-- TODO: Automate IEEEPARstart -->
platforms allow people to express their opinions freely, and stimulate collaboration across the globe. Unfortunately, online interaction may often come with loosened inhibitions in making profane, bigoted, or offensive remarks. We refer to such unwelcome remarks as "toxic chat". Online systems may or may not have their own embedded profanity filtering, and those that do typically use pre-registered terms and simple pattern matching. This approach lacks the deeper contextual understanding needed to identify sentences that are toxic but that may not contain banned terms. Thus we propose a new toxic chat filtering system that differentiates itself in that a) its filtering is based on machine learning and deeper contextual analysis, and b) it is deployed as a scalable and easily integrated web framework that can be adapted to any source of text for online interaction of any size. The platform is based on Docker and Kubernetes for easy deployment and dependency management and to allow for fast scale-out to large systems. It uses state-of-the-art distributed systems technology for processing and storage, to allow for rapid scaling to any size while maintaining a shared file space (HDFS) between each Kubernetes Zone. This paper presents the architecture, development, and use of this system in the context of a web chat application and Twitch chatBot as motivating examples.

## What we are going to make 

We will create a prototype of PaaS/SaaS service that provides the following specific capabilities:

* machine-learning-based toxic chat identification and filtering engine
* integerated web chat application or chatbot that uses the engine to analyze a real-time stream of text

Toxic Comment Classifier
========================

## Training data

There is a dearth of labelled datasets for training classifiers to detect toxic comments. We conducted an online search and literature review on IEEE Xplore, Scopus, and Science Direct and concluded that the best dataset available is Toxic Comment Classification Challenge dataset released by Jigsaw and Google on Kaggle in 2018 https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
 [@jigsaw2018]

Example of a single citation [@rybinski2018]. 
Example of a two citations [@salminen2018;@nestor2018]. 
Example of multiple citation [@loi2018;@spiros2018;@david2018].


## Machine Learning Algorithm

TBD

Cloud system integration and development
========================================

The following technologies and solutions were integrated to provide a general framework that can scale to high volume/traffic in the future.

* Docker, Kubernetes - easy deploy and scale out
* CI/CD pipeline - to automate the build, integration, and deployment
* AWS, GCP, Heroku, etc - to deploy the solution into a mainstream PaaS infrastructure 
* HDFS or similar - to store big data and share it between systems
* Scikit-learn or Apache Spark + MLlib - machine learing for the detox engine
* RESTful APIs - to help other applications integrate detox engine in the system

## Deployment and scaling
* Docker, Kubernetes - easy deploy and scale out

## Automated build, integration, deployment
* CI/CD pipeline - to automate the build, integration, and deployment

## IaaS/PaaS infrastructure
* AWS, GCP, Heroku, etc - to deploy the solution into a mainstream PaaS infrastructure 

## Shared storage
* HDFS or similar - to store big data and share it between systems

## Machine Learning Framework
* Scikit-learn or Apache Spark + MLlib - machine learing for the detox engine

## Application Programming Interfaces
* RESTful APIs - to help other applications integrate detox engine in the system





Acknowledgment {#acknowledgment .unnumbered}
==============

The authors would like to thank ...

References {#references .unnumbered}
==========

