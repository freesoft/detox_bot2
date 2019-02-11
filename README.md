# detox_bot2

It's for UIUC MCS-DS CS498 CCA project, upgrading and expending the previous [detox_bot](https://github.com/freesoft/detox_bot) project to use more Cloud stuff that we've learned in CS498 CCA class in Spring 2019.

Pleae read the original project's README file to have some understanding for what is it and what we can improve.

A few items that I think we can improve. Added words with bold at the end of each item that I think it's related to CCA topics.

* Since it's already dockerized, maybe run it somewhere in AWS(maybe Amazon ECS?) **AWS**, **ECS**, **Docker**, **Virtualization**
* It has classifier + web chat ( or classifier + Twitch TV ChatBot ) combined togehter in the project. We could break it down and make it as separate project/docker module. **Docker**, **Virtualization**
* The project is dockerized, we could make it Kunbernetes Pod and deploy on tghe Kubernetes Cluster for easiler scale out. **Kubernetes**, **Docker**, **Virtualization**
* The project create a trained model and use it for toxic chat prediction. Unfortunately, as training data set goes bigger and bigger, so does this trined model. For best performance, we need to serialize it and store in the file( or any persistent layer ), so that next time when app is running we just need to load it to the memory, no need to worry about re-training. We could use HDFS instead of local filesystem in the instance? **HDFS**, **AWS EBS**, **Big Data**
* Related to above one. The project uses Python's scikit-learn for training. Probably we can upgrade it to use Spark(PySpark or with Scala) so that it's much faster and cloud-ready? **Spark**, **Big Data**, **RDD**
* The project has simplye Python based Flask-websocket based webchat app. It has ugly UI design and so does functionality since I just created it in an hour by copying existing Flask-websocket example from web search and modified a bit. We could improve this webchat app ( or even just replace it with something existing and customizable ) so our app look better and great. **Websocket**, **REST**, **HTTP**, **RPC**
