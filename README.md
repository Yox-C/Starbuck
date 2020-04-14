Starbuck

Installation
      
      Anaconda distribution of Python version 3.6 or later

Project Motivation
      
      To improve the understanding of the behaviours better,it is important to analyze the histrical beahaviour.Fit the model based on the history to predict the advertising success rate.

Python Libraries used
      
      pandas for data munipulation
      numpy for numbers crunching
      math for simple math operations
      json for reading json files.
      matplotlib.pyplot and seaborn for data visualization
      sklearn for machine learning algorithms
      
Dataset

The dataset is in folder data, contained in three files:

portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers completed Here is the schema and explanation of each variable in the files:
portfolio.json

id (string) - offer id
offer_type (string) - type of offer ie BOGO, discount, informational
difficulty (int) - minimum required spend to complete an offer
reward (int) - reward given for completing an offer
duration (int) -
channels (list of strings)
profile.json

age (int) - age of the customer
became_member_on (int) - date when customer created an app account
gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
id (str) - customer id
income (float) - customer's income
transcript.json

event (str) - record description (ie transaction, offer received, offer viewed, etc.)
person (str) - customer id
time (int) - time in hours. The data begins at time t=0
value - (dict of strings) - either an offer id or transaction amount depending on the record

结果讨论
三种方法分别是bagging,randomForest,adaboost方法。从评估结果的信息来看，adaboost方法的准确性更高，该集成方法是将使用的给定的学习算法构建的基本分类器的预测结合起来，以此来提高分类器的通用性\鲁棒性。在该种情况下我们可以使用gridsearchcv方法来对参数进行进一步的筛选。

使用网格搜索得出了最优参数{'learning_rate': 0.2, 'n_estimators': 300}。

限于时间和机器性能，没有进行更多尝试，后续可以尝试更多不同算法，以比对时间花销和精度。

后续可以继续研究顾客使用优惠券的时间与用户本身信息的关系，可以得出更加有效的优惠券有效期。
