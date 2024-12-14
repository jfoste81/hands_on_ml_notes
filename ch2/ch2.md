<font size=6>**Chapter 2 - End-to-End Machine Learning Project**</font>

<font size=4>**Table of Contents**</font>

- [Working with Real Data](#working-with-real-data)
- [Look at the Big Picture](#look-at-the-big-picture)
  - [Frame the Problem](#frame-the-problem)
      - [Pipelines](#pipelines)
  - [Select a Performance Measure](#select-a-performance-measure)
  - [Get the Data](#get-the-data)
  - [Create a Test Set](#create-a-test-set)


# Working with Real Data

- In this chapter we will be using the California Housing Prices dataset StatLib repo. 

![California Housing Prices Figure](images/2_1.png)

# Look at the Big Picture 

- First task is to use California census data to build a model of housing prices in the state, including metrics such as popualtion, median income, and median housing price for each block group in California. 
  - Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (typically $600$ to $3,000$ people)
    - **districts** for short
- **Machine Learning Checklist:** 

    ![Machine Learning Project Checklist](images/app_A.png)

## Frame the Problem

* From a corporate perspective, first you might consider the business objective
  * How can the company use and benefit from this model?
* Knowing this objective is important, determines how you frame the problem, which algorithms to select, which performance measure you will use to evaluate the model, and how much effort will be spent tweaking

* In our case, our model's output (prediction of district's median housing price) will be fed to another machine learning system along with other **signals**, pieces of info fed to a machine learning system. 
![Machine Learning Pipeline](images/2_2.png)

* This downstream system will determine if it is worth investing into a given area. 

#### Pipelines

* A sequence of data processing components is called a **data pipeline**.
  * Pipelines are common in ML systems, since there is a lot of data to manipulate and many data transformations to apply.

* Components typically run asynchronously. Each component pulls in a large amount of data, processes it, and spits out the result in another data store. 
  * Later, the next component in the pipeline pulls in the data and spits out its own output. 
  * Each component is fairly self-contained: the interface between components is simply the data store. 
  * This simplifies the system. If a component breaks down, the downstream components can often continue to run normally by just using the last output from the broken component. 
    * This makes the architecture very robust. 
    * However, this also means broken components can go unnoticed for some time if not monitored closely. The data will get stale and the overall system performance will drop.
* This is a **supervised**, **multiple regression** model (since the system will use multiple features to make a prediction).
  * It is also a **univariate regression** problem, since we are only trying to predict a single value for each district. 
    * If it were multiple values, it would be **multivariate regression**.
* Since there is no continuous flow of data coming in, it is a simple **batch learning** problem. 

## Select a Performance Measure

* Now need to select performance measure. Regrssion problems typically use the **root mean square error**.
  * Gives an idea of how much error the system makes in its predictions, with higher weight given to large errors.

* Equation for RMSE: 
$$
RMSE \ ( \ X \ , \  h \ )  \ =  \ 1  \ m \  \sum \ i=1 \ m \ h (x  \ (i) \  )-y  \ (i) \ 2
$$
  * This equation introduces several common notations for this text => 
    * $m$ is the number of instances in the dataset you are measuring RMSE on =>
      * For example, if you evalauate the RMSE on a validation set of 2,000 districts, then $m=2,000$
    * $\mathbf{x}^{(i)}$ is a vector of all feature values (excluding label) of the $i^{th}$ instance in the dataset, and $y^{(i)}$ is its label (desired output value for that instance)  
      * For example, if first district in dataset is located at longitutde $-118.29\degree$, latitude $33.91\degree$, and it has $1,416$ inhabitants with a median income of $\$38.372$, and the median house value is $\$156,400$ (ignoring other features for now), then =>
        $$
        x \ (1) \ = \ [-118.29, \ 33.91, \ 1,416, \ 38,372] 
        \\
        and: 
        \\
        y \ (1) \ = \ 156,400
        $$
    * $\mathbf{X}$ is a matrix containing all feature values (excluding labels) of all instances in the dataset. There is one row per instance, and the $i^{th}$ row is equal to the transpose of $\mathbf{x}^{i}$, noted $(\mathbf{x}^{i})^T$ 
    $$
    X \ = \ [x(1))^T , x(2)^T : x(1999)^T, x(2000)^T] = [-118.29, \ 33.91, \ 1,1416, \ 38,372 : \ : \ : \ ]
    $$
    * $h$ is your system's prediction function, also called a **hypothesis**. WHen your system is given an instance's feature vector $\mathbf{x}^{(i)}$, it outputs a predicted value $ŷ^{(i)} \ = \ h(\mathbf{x}^{(i}))$
      * $ŷ$ is pronounced "y-hat"
      * For example, if your system predicts that the median housing price in the first district is $\$158,400$, then $ŷ^{(i)} \ = \ h(\mathbf{x}^{(i})) = 158,400$. The prediction error for this district is $ŷ^{(1)} \  - \ y^{(1)} = 2,000$
    * $RMSE(\mathbf{X},h)$ is the cost function measured on the set of examples using your hypothesis $h$
* We use lowercase italics for scalar values (such as $m$ or $y^{(i)}$) and function names (such as $h$), lowercase bold font for vectors (such as $\mathbf{x}^{(i)}$), and uppercase bold font for matrices (such as $\mathbf{X}$)

* Although the RMSE is usually preferred in regression tasks, sometimes you might use other functions. If there are many outliers, maybe you oonsider the **mean absolute error** (MAE, also called **average absolute deviation**)
  * $MAE \ ( \ X \ , \ h \  ) \ = \ 1 \ m \sum \ i=1 \ m \ h \ ( \ x \ (i) \ ) \ - \ y \ (i)$
* Both the RMSE and the MAE are ways to measure the distance between two vectors: the vector of predictions and the vector of target yields. Various distance measures, or **norms**, are possible. 
  * Computing the root of a sum of squares (RMSE) corresponds to the **Euclidean Norm**
    * A notion of distance, also called the $\ell_2 \ norm$\, noted $|| \cdot ||_2$ (or just $|| \cdot ||$)
    * Computing the sum of absolutes (MAE) corresponds to the $\ell_1 \ norm$, noted $|| \cdot ||_1$. This is also sometimes called the **Manhattan Norm** because it measures the distance between two points in a city if you can only travel along orthogonal city blocks. 
    * More generally, the $\ell_k \ norm$ of a vector $\mathbf{v}$ containing $n$ elements is defined as 
     $$
     \mathbf{|| v ||}_k \ = \ (|v_1|^k \ + |v_2|^k \ + \ ... \ + \ |v_n|^k)^{1/k} \cdot \ell_0
     $$
    * This gives the number of nonzero elements in the vecotr, and $\ell_\infty$ gives the maximum aboslute value in the vector
* The higher the norm index, the more it focuses on large values and neglects small ones. This is why the RMSE is more sensitive to outliers than the MAE. But when outliers are exponentially rare (in a bell-shaped curve for instance), the RMSE is generally preferred. 

## Get the Data

* Will be working in *02_end_to_end_machine_learning_project* notebook

* load_housing_data() will look for the datasets/housing.tgz file and will create the datasets directory inside the current directory if it cannot find it.
* It will then download the dataset from the ageron/data GitHub repo (referenced from the url) and extract its contents into the datasets directory with a .csv file
* Lastly, the function loads the CSV file into a Pandas DataFrame object and returns it

* Pandas head() method returns the top five rows of data
* Each row represents one district. There are 10 attributes: longitutde, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, and ocean_proximity. 

* The info() method is useful to get a quick description of the data, in particular the total number of rows, each attribute's type and the number of non-null values: 

![info method](../images/info().png)

* total_bedrooms has 20,433 non-null values out of the 20,640 total. That means that there are 207 districts that are missing and that will need to be fixed later. 
* Using context clues and its repetitive nature, we can deduce that the ocean_proximity attribute is likely a categorical text attribute. We can use the value_counts() method to detect how many districts belong to each category. 
* The describe() method shows a summary of numerical attributes.

## Create a Test Set