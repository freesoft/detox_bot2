# detox_bot2

## Branching strategy - Please Read before working on branch

* The project uses GitFlow branching strategy. If you are not familiar to this, read https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
* TL;DR; always create your feature branch from develop branch, work on your feature branch and test, when you think your feature branch is good to deploy to production, then merge to develop branch. Develop branch should always be "ready to production" status. When we deploy, we are going to create new release branch from develop branch, make artifact(or docker image) and deploy to production, and merge the release branch to master branch.

## Important Key Dates

### proposal
* https://piazza.com/class/jqz0r68mx9863m?cid=167 Friday of week 6. It has been fixed now on the course project overview page.

## General idea for the project
It's for UIUC MCS-DS CS498 CCA project, upgrading and expending the previous [detox_bot](https://github.com/freesoft/detox_bot) project to use more Cloud stuff that we've learned in CS498 CCA class in Spring 2019.

Pleae read the original project's README file to have some understanding for what is it and what we can improve.

A few items that I think we can improve. Added words with bold at the end of each item that I think it's related to CCA topics.

* Since it's already dockerized, maybe run it somewhere in AWS(maybe Amazon ECS?) **AWS**, **ECS**, **Docker**, **Virtualization**
* It has classifier + web chat ( or classifier + Twitch TV ChatBot ) combined togehter in the project. We could break it down and make it as separate project/docker module. **Docker**, **Virtualization**
* The project is dockerized, we could make it Kunbernetes Pod and deploy on tghe Kubernetes Cluster for easiler scale out. **Kubernetes**, **Docker**, **Virtualization**
* The project create a trained model and use it for toxic chat prediction. Unfortunately, as training data set goes bigger and bigger, so does this trined model. For best performance, we need to serialize it and store in the file( or any persistent layer ), so that next time when app is running we just need to load it to the memory, no need to worry about re-training. We could use HDFS instead of local filesystem in the instance? **HDFS**, **AWS EBS**, **Big Data**
* Related to above one. The project uses Python's scikit-learn for training. Probably we can upgrade it to use Spark(PySpark or with Scala) so that it's much faster and cloud-ready? **Spark**, **Big Data**, **RDD**
* The project has simple Python based Flask-websocket based webchat app. It has ugly UI design and so does functionality since I just created it in an hour by copying existing Flask-websocket example from web search and modified a bit. We could improve this webchat app ( or even just replace it with something existing and customizable ) so our app look better and great. **Websocket**, **REST**, **HTTP**, **RPC**

## Final project and report related questions

https://piazza.com/class/jqz0r68mx9863m?cid=176

Q) In the first 5 weeks of the class, we cover a wide range of divergent but tangentially related topics.  Can we concentrate on one technology for our project?  Such as MapReduce?  Spark w/ML?  Load balancing a highly available application?<br/>
A) Yes you can


Q) Is there an expectation of final “transformation” of data?<br/>
A) Not really (I'm not entirely clear what you mean by final transormation).


Q) Can we simply filter some data or must some greater correlation take place?<br/>
A) You must have some novel findings or can even replicate, match, or beat existing work out there. So just filtering data is not good enough. 


Q) Are we expected to present the transformed data in a particular format? <br/>
A) No


Q) We know you want a “report” but should you have access to the final data? <br/>
A) No 


Q) What is expected in the final report?  We assume you may not expect to  have access to our actual data (especially if we use the provided AWS access) and can only rely on what is provided in that report.  Architecture diagrams?  Class diagrams?<br/>
A) The final report should be like a mini research paper where you write about your motivation, approach, findings, dataset, where you got stuck, etc. If you look at the each milestone's requirement, it should become very clear.



