---
title: "Practical Machine Learning - Predictive Modeling Course Project"
author: "Camelia Guild"
date: "9/10/2021"
output:
  html_document: 
    keep_md: yes
---


**Background**: Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

**Objective**: Our goal will be to predict qualitative activity recognition of weight lifting exercises using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways:exactly according to the specification (Class A), throwing elbows to the front (Class B), lifting 
the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D), and throwing the hips to the front (Class E).

The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Data source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

#### Download the file from the web.


```r
# Check for and create a directory if it doesn't already exist
#if (!file.exists("Project")){
#    dir.create("Project")
# }
# Download file from the web
#trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#download.file(trainUrl, destfile = "./pml-training.csv", method="curl")

#testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download.file(testUrl, destfile = "./pml-testing.csv", method="curl")
#list.files("./Project")

data.temp <- read.csv("./Project/pml-training.csv", header = TRUE, na.strings=c("NA","#DIV/0!",""))
test.temp <- read.csv("./Project/pml-testing.csv", header = TRUE, na.strings=c("NA","#DIV/0!",""))

dim(data.temp)
```

```
## [1] 19622   160
```

```r
dim(test.temp)
```

```
## [1]  20 160
```


#### Load required libraries

```r
library(dplyr)
library(rpart)
library(caret)
library(naniar)
library(corrplot)
```


# Data Pre-processing and Feature Selection

This section involves (1) checking that both data sets contain the same predictor variables. (2) Removing irrelevant variables. (3) Removing variables with mostly missing data. (4) Removing any near-zero variance variables.

#### Checking for common predictors.
The results show that data.clean has the variable "classe" and test.clean contains "problem_id". Thus, problem_id will be removed from the test set.

```r
# Get the colnames in each data set
df1 <- colnames(data.temp)
df2 <- colnames(test.temp)
# Check that the two data sets contain the same colnames. A return of False means that they do not contain the same named columns.
all(df1 %in% df2)
```

```
## [1] FALSE
```

```r
# Get the difference in colnames between the two data sets.
# The variable, classe, is in the full data set but not in the test set
setdiff(df1, df2)
```

```
## [1] "classe"
```

```r
# The variable, problem_id,is in the test data set but not in the full data set.
setdiff(df2, df1)
```

```
## [1] "problem_id"
```

#### Removing irrelevant variables
Irrelevant variables have no predictive information and should be removed from the data.


```r
# The following irrelevant variables appear in both data sets and must be removed. In addition, problem_id which only appears in the test data set must be removed since both data sets must contain the same predictors.
names(data.temp[,1:5])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"
```

```r
data.clean <-data.temp[,-c(1:5)]
test.clean <-test.temp[,-c(1:5,160)]
dim(data.clean)
```

```
## [1] 19622   155
```

```r
# Test has 1 less variable than the full data set; it does not include "classe". We will classify "classe" later in a final step.
dim(test.clean) 
```

```
## [1]  20 154
```


#### Removing variables containing a substantial number of NA (missing) values
There are 100 columns in both data sets which contain all or mostly NA (missing) values, ranging from 97% to 100% missing. See the Appendix for a more complete breakdown.


```r
# Sum up the missing values in each variable
sumNA <- apply(apply(data.clean,2,is.na),2,sum) %>% as.data.frame
#summary(sumNA)

sumNA <- apply(apply(test.clean,2,is.na),2,sum) %>% as.data.frame
#summary(sumNA)

# Remove missing data from the training data set.
manyNA <- sapply(data.clean, function(x) mean(is.na(x))) > 0.95
data.clean2 <- data.clean[, manyNA==FALSE]
dim(data.clean2)
```

```
## [1] 19622    55
```

```r
# Remove missing data from the test data set
manyNA <- sapply(test.clean, function(x) median(is.na(x))) >0.95
test.clean2 <- test.clean[, manyNA==FALSE]
dim(test.clean2)
```

```
## [1] 20 54
```

```r
# Check that there are no missing values in the data
#sum(is.na(data.clean))
#sum(is.na(test.clean))
```

#### Removing near-zero variance predictors

Near-zero variance predictors are variables that might have only a handful of unique values that occur with very low frequencies. That is, they may have a single value for the vast majority of the observations.


```r
nZeroVar <- nearZeroVar(data.clean2, saveMetrics = TRUE)
#head(nZeroVar)
nZeroVar <- nearZeroVar(test.clean2, saveMetrics = TRUE)
#head(nZeroVar)

# new_window is near zero variance and will be removed from both data sets.
 data.clean2 <- data.clean2[,-1]
 test.clean2 <- test.clean2[,-1]
```

#### The cleaned data sets
The full data set contains 19,622 observations and 54 variables. The test set contains 20 observations and 53 variables.


```r
# The final data sets to be used in the predictive modeling. The full data set contains
# 54 variables and the test data set contains 53. Remember that the test data set has 
# 1 less variable than the full data set since it does not contain the outcome variable "classe".
 dim(data.clean2)
```

```
## [1] 19622    54
```

```r
 dim(test.clean2)
```

```
## [1] 20 53
```

# Exploratory Data Analysis
The table below shows the distribution of the dependent variable, classe. Class A has the highest occurrence and class D the lowest.


```r
freq <- table(data.clean2$classe)
Percent <- paste(round(prop.table(freq)*100, 2), "%", sep="")
class_freq <- as.data.frame(rbind(Freq=round(freq), Percent))
knitr::kable(
    class_freq[1:2, 1:5], caption = 'Descriptive Statistics of the Exercise Classification Variable')
```



Table: Descriptive Statistics of the Exercise Classification Variable

|        |A      |B      |C      |D      |E      |
|:-------|:------|:------|:------|:------|:------|
|Freq    |5580   |3797   |3422   |3216   |3607   |
|Percent |28.44% |19.35% |17.44% |16.39% |18.38% |

The correlation plot below shows the relationships between the predictors. Dark blue colors indicate strong positive correlations, and dark red indicates stong negative correlations. In this plot, the predictor variables have been grouped using a clustering technique so that collinear groups of predictors are adjacent to one another. See the Appendix for the correlation values for the highly correlated predictors.


```r
corr_df <-cor(data.clean2[,-54])
corrplot(corr_df, type="upper", method="circle", order="hclust",tl.cex=0.6, tl.col ="black")
```

![](index_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

### Splitting the data 
The data is split into a training set and validation set.

```r
set.seed(12359)
inTrain <- createDataPartition(data.clean2$classe, p=0.70, list=FALSE)
trainData <- data.clean2[inTrain, ]
validData <- data.clean2[-inTrain, ]

# set x and y to avoid slowness of caret() with model syntax
y<- trainData[,54]
x<- trainData[,-54]

library(parallel)
library(doParallel)
# convention to leave 1 core for OS
cl <- makeCluster(detectCores() - 1) 
registerDoParallel(cl)
```

# Predictive Modeling with Bagging Trees
In bagging, we build a number of decision trees on bootstrapped training samples. When building decision trees, the algorithm considers a majority of the predictors as split candidates. That is, most or all predictors are considered before a decision is made on which predictor to use for a split. This process is done for each split.


```r
# Model: Bagging Trees
#set.seed(12359)
# The 5-fold cross-validation specification splits the training data into
# 5 train/test sets. This allows us to use multiple test sets, thereby
# averaging out-of-bag error.
control = trainControl(method="cv", number=5, allowParallel = T)
bag.tree <- train(as.factor(classe) ~ ., data=trainData, method="treebag", trControl=control)
bag.tree
```

```
## Bagged CART 
## 
## 13737 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10990, 10989, 10989, 10990 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.9906093  0.9881206
```

The bagged trees model accuracy is 99.06% with 0.94% OOB error rate. This is approximately what we would expect to get when our bagged tree model is applied to our validation/test set. However, although we took steps to lessen the problem of over-fitting our training data by using cross-validation, over-fitting still remains a problem because we have matched our algorithm to the data. Therefore, we would expect our out of sample accuracy to be lower and the error to be higher when our built model is applied to our validation/test set.

#### Prediction estimate for out of sample accuracy

Next, we apply our bagged trees model to the validation data set.

```r
bag.predValid <- predict(bag.tree, validData)
confusionMatrix(factor(validData$classe), bag.predValid)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    2    1    0    0
##          B   17 1117    4    1    0
##          C    4    2 1015    5    0
##          D    3    2   17  941    1
##          E    4    1    4    4 1069
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9878          
##                  95% CI : (0.9846, 0.9904)
##     No Information Rate : 0.2887          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9845          
##                                           
##  Mcnemar's Test P-Value : 0.0001255       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9835   0.9938   0.9750   0.9895   0.9991
## Specificity            0.9993   0.9954   0.9977   0.9953   0.9973
## Pos Pred Value         0.9982   0.9807   0.9893   0.9761   0.9880
## Neg Pred Value         0.9934   0.9985   0.9946   0.9980   0.9998
## Prevalence             0.2887   0.1910   0.1769   0.1616   0.1818
## Detection Rate         0.2839   0.1898   0.1725   0.1599   0.1816
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9914   0.9946   0.9864   0.9924   0.9982
```

# Predictive Modeling with Random Forest
Random forests provide an improvement over bagged trees by decorrelating the trees; that is, it forces each split to consider only a random subset of the predictors. Because of this, random forests will typically be helpful when we have a large number of correlated predictors.Thus, random forests leads to a reduction in both test error and OOB error. 


```r
#set.seed(12359)
control = trainControl(method="cv", number=5, allowParallel = T)
rf.tree <- train(x,y, data=trainData, method="rf", trControl=control)
# The tuneLength parameter tells the algorithm to try a specified number of levels values for the tuning parameters. TuneLength = 3 is the default. 
rf.tree <- train(as.factor(classe) ~ ., data=trainData, method="rf", trControl=control, verbose=FALSE)
rf.tree
```

```
## Random Forest 
## 
## 13737 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10991, 10989, 10989, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9943220  0.9928171
##   27    0.9971610  0.9964088
##   53    0.9926477  0.9907002
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

A look at the 5-fold cv. The average of the 5 performance estimates would be the cross-validation estimate of model performance. Therefore, our observed model accuracy is 99.72% with 0.28% out-of-bag (OOB) error rate. This is approximately what we would expect to get when applied to our validation/test set. Again, although we used cross-validation, over-fitting still remains a problem because we have matched our algorithm to the data. Therefore, we would expect our out of sample accuracy to be lower and the error to be higher when our built model is applied to our validation/test set.


```r
# Look at the 5-fold cv resample
rf.tree$resample
```

```
##    Accuracy     Kappa Resample
## 1 0.9970867 0.9963147    Fold1
## 2 0.9978166 0.9972382    Fold3
## 3 0.9959971 0.9949364    Fold2
## 4 0.9967249 0.9958568    Fold5
## 5 0.9981798 0.9976978    Fold4
```

```r
confusionMatrix(rf.tree)
```

```
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.3  0.0  0.0  0.0
##          C  0.0  0.0 17.4  0.1  0.0
##          D  0.0  0.0  0.0 16.3  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9972
```


#### Prediction estimate for out of sample accuracy
Next, we apply our random forest model to the validation set.

```r
# Use the rf.tree model to predict the test set
rf.predValid <- predict(rf.tree, validData)
confusionMatrix(factor(validData$classe), rf.predValid)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    1
##          B   11 1127    1    0    0
##          C    0    4 1022    0    0
##          D    0    0    6  958    0
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9954         
##                  95% CI : (0.9933, 0.997)
##     No Information Rate : 0.2862         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9942         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9935   0.9965   0.9932   0.9958   0.9991
## Specificity            0.9998   0.9975   0.9992   0.9988   0.9992
## Pos Pred Value         0.9994   0.9895   0.9961   0.9938   0.9963
## Neg Pred Value         0.9974   0.9992   0.9986   0.9992   0.9998
## Prevalence             0.2862   0.1922   0.1749   0.1635   0.1833
## Detection Rate         0.2843   0.1915   0.1737   0.1628   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9966   0.9970   0.9962   0.9973   0.9991
```

## Results
Applying our bagged trees model and random model to the validation data set, we were able to obtain a prediction accuracy and out of sample error rate for each model. For the bagged trees model, our prediction accuracy is 98.78% with an out of sample error of 1.22%. For the random forests model, our prediction accuracy is 99.54% with an out of sample error of 0.46%. This confirms that our random forest model has a lower OOB error and test error. Therefore, the random forests model is an improvement in predictive performance. Next, we will apply the random forests model to the 20-sample test data.


#### Prediction estimate for the 20-sample test cases


```r
# Prediction on the 20 sample test set
rf.pred.test <- predict(rf.tree, newdata=test.clean2)
rf.pred.test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



### References

For a parallel implementation of random forest with caret::train function tutorial:
https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md

An Introduction to Statistical Learning: with Applications in R by Gareth James, Daniela Witten, Trevor Hastie and Rob Tibshirani. 2017 edition.


\newpage
## Appendix
#### Missing data analysis

```r
# There are 100 predictors with a substantial amount of missing data in each. Missing data range from 97% to 100% of missing values.
miss_var_table(data.clean)
```

```
## # A tibble: 17 x 3
##    n_miss_in_var n_vars pct_vars
##            <int>  <int>    <dbl>
##  1             0     55   35.5  
##  2         19216     67   43.2  
##  3         19217      1    0.645
##  4         19218      1    0.645
##  5         19220      1    0.645
##  6         19221      4    2.58 
##  7         19225      1    0.645
##  8         19226      4    2.58 
##  9         19227      2    1.29 
## 10         19248      2    1.29 
## 11         19293      1    0.645
## 12         19294      1    0.645
## 13         19296      2    1.29 
## 14         19299      1    0.645
## 15         19300      4    2.58 
## 16         19301      2    1.29 
## 17         19622      6    3.87
```

```r
miss_var_table(test.clean)
```

```
## # A tibble: 2 x 3
##   n_miss_in_var n_vars pct_vars
##           <int>  <int>    <dbl>
## 1             0     54     35.1
## 2            20    100     64.9
```

#### Correlation coefficients for highly correlated predictors

```r
corr_df <-cor(data.clean2[,-54])
#corrplot(corr_df, type="lower", method="circle", order="hclust", tl.cex=0.6, tl.col ="black")
 for (i in 1:nrow(corr_df)){
     correlations <- which((corr_df[i,] > 0.85) & (corr_df[i,] !=1))
     if(length(correlations) >0){
         #print(colnames(data.clean2)[i])
         #print(correlations)
     }
 }
myvars <- c("gyros_dumbbell_z", "gyros_dumbbell_x", "gyros_forearm_z", "accel_belt_y", "roll_belt","total_accel_belt", "accel_belt_z", "gyros_arm_x", "gyros_arm_y", "accel_belt_x","magnet_belt_x","pitch_belt","yaw_dumbbell", "accel_dumbbell_z")
df <- data.clean2[myvars]
hCorr_df <-cor(df)
corrplot(hCorr_df, type="upper", method="number", order="hclust", tl.cex=0.6, tl.col ="black", number.cex=0.6)
```

![](index_files/figure-html/unnamed-chunk-19-1.png)<!-- -->

#### Random forest predictors selection
Random forest algorithm selects the subset of variables at each split that produces the highest OOB prediction accuracy.

```r
# Plot of randomly selected predictors by cv-accuracy
plot(rf.tree)
```

![](index_files/figure-html/unnamed-chunk-20-1.png)<!-- -->

#### Plot of predictors by magnitude of importance

```r
# Get a plot and list of variables ranked according to importance
#print(importance)
importance <- varImp(rf.tree, scale=F)
plot(importance, main="Variable Importance - Top 10", top=10)
```

![](index_files/figure-html/unnamed-chunk-21-1.png)<!-- -->
