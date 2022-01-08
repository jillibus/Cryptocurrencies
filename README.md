# Cryptocurrencies

![logo](images/module_18_logo.png)

# Overview
          
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, a bank or lending company will need to employ different techniques to train and evaluate models with unbalanced classes. 

In this project, I will be using several different methods to calculate Credit Risk and return which algorithm was the best at Predicting the Credit Risk of the Data Set I chose.  I will be using a credit card dataset from LendingClub, a peer-to-peer lending services company.  I will use tools to _Over Sample_ the data, _Under Sample_ the data, using machine learning techniques to predict whether the algorithm, with the data set chosen, produces a high percentage of true outcomes. 

For Deliverable 1, I will use Resampling Models to Predict Credit Risk.

For Deliverable 2, I will use the SMOTEENN Algorithm to Predict Credit Risk.

For Deliverable 3, I will use the Ensemble Classifiers to Predict Credit Risk.

The results of the above, will allow me to provide an analysis of which machine language algorithm worked best for Predicting the Outcome for Credit Risk. 

<img src="images/ml_algorithms.png" />

# Resources
* Data Sources: LoanStats_2019Q1.csv
* Software: Jupyter Notebook, Python 3.7, Pandas

# GitHub Application Link

<a href="https://jillibus.github.io/Cryptocurrencies">Cryptocurrencies</a>

## Deliverable 1: Preprocessing the Data for PCA

Data preprocessing involves transforming raw data to well-formed data sets so that data mining analytics can be applied. Raw data is often incomplete and has inconsistent formatting. The adequacy or inadequacy of data preparation has a direct correlation with the success of any project that involve data analyics.

Preprocessing involves both data validation and data imputation. The goal of data validation is to assess whether the data in question is both complete and accurate. The goal of data imputation is to correct errors and input missing values -- either manually or automatically through business process automation (BPA) programming.

Data preprocessing is used in both database-driven and rules-based applications. In machine learning (ML) processes, data preprocessing is critical for ensuring large datasets are formatted in such a way that the data they contain can be interpreted and parsed by learning algorithms. (https://www.techopedia.com/definition/14650/data-preprocessing)

In our project we performed _Data Preprocessing_ by perfomring the following steps:
*  In the **crypto_df** 
  *  All cryptocurrencies that are not being traded are removed.
  *  The IsTrading column is dropped
  *  All the rows that have at least one null value are removed.
  *  All the rows that do not have coins being mined are removed.
  *  The CoinName column is dropped.
* A new DataFrame is created that stores all cryptocurrency names from the CoinName column and retains the index from the crypto_df DataFrame.
* The get_dummies() method is used to create variables for the text features, which are then stored in a new DataFrame, X.
* The features from the X DataFrame have been standardized using the StandardScaler fit_transform() function.



## For Deliverable 2: Reducing Data Dimensions Using PCA
Using the Principal Component Analysis (PCA) Algorithm, I will reduce the dimensions of the X DataFrame to 3 principal components and place these dimensions into a new DataFrame.

In our project we created a new DataFrame **pcs_df** that added new columns, _PC 1_, _PC 2_, _PC 3_, and used the index of **crypto_df** DataFrame as the index.

<img src=            />




## For Deliverable 3: Clustering Cryptocurrencies Using K-means
Using the K-means algorithm, I will create an _Elbow Curve_ using the _hvPlot_ to find the best value for _K_ from the **pcs_df** DataFrame.  Once that is found, I will use the K-means algorithm and use the K value found to predict the number of K clusters for the cryptocurrencies' data.

* I will create a new DataFrame, **clustered_df**, which concatenates the **crypto_df** with the **pcs_df**, the index will be the same as the **crypto_df**.
* I will then add the CoinName column from the **cryptonames_df** to the **clustered_df**.
* Lastly, I will add a new column, **clustered_df['Class']** which holds the predictions (model.labels_)

<img src=          />


## For Deliverable 4: Visualizing Cryptocurrencies Result
For this deliverable, I will demonstrate, visually, using scatter plots, the distinct groups that correspond to the 3 principal components (PC 1, PC 2, PC 3) as well as a table with all of the currently tradable cryptocurrencies using the _hvplot.table()_ function.

<img src=             />

<img src=            />

# Summary


Thank you for your time and let me know if you wish to see any additional data.

Jill Hughes
