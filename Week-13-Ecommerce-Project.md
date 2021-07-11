---
title: "Week 13 Ecommerce Project"
author: "Mr. Mutai"
date: "7/11/2021"
output: 
  html_document: 
    keep_md: yes
---




# ASSESSMENT QUESTION 

Kira Plastinina is a Russian brand that is sold through a defunct chain of retail stores in Russia, Ukraine, Kazakhstan, Belarus, China, Philippines, and Armenia. The brand’s Sales and Marketing team would like to understand their customer’s behavior from data that they have collected over the past year. More specifically, they would like to learn the characteristics of customer groups.

# 1. DEFINING THE QUESTION 

## i) Defining the Specific Data Analytic Question

The brand's Sales and Marketing team would like to understand  the characteristics of customer groups in the respective countries.

## ii) Defining the Metric for Success

To be able to build unsupervised learning algorithms that will help us understand the characteristics of customer groups in our dataset. 

## iii) Understanding the Context

Kira Plastinina is a Russian fashion designer and entrepreneur whose first brand store opened in Moscow, 2017. The brand has been worn by many celebrities such as Paris Hilton and Lindsay Lohan. 

## iv) Recording the Experimental Design

1. Problem Definition
2. Data Sourcing
3. Check the Data
4. Perform Data Cleaning 
5. Perform Exploratory Data Analysis  (Univariate, Bivariate & Multivariate)
6. Implement the Solution using the unsupervised learning algorithms.
7. Challenge the Solution
8. Follow up Questions


# 2. IMPORTING THE RELEVANT LIBRARIES


```r
library(data.table)
library(ggplot2)
library(caret)
```

```
## Loading required package: lattice
```

```r
library(caretEnsemble)
```

```
## 
## Attaching package: 'caretEnsemble'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     autoplot
```

```r
library(psych)
```

```
## 
## Attaching package: 'psych'
```

```
## The following objects are masked from 'package:ggplot2':
## 
##     %+%, alpha
```

```r
library(Amelia)
```

```
## Loading required package: Rcpp
```

```
## ## 
## ## Amelia II: Multiple Imputation
## ## (Version 1.8.0, built: 2021-05-26)
## ## Copyright (C) 2005-2021 James Honaker, Gary King and Matthew Blackwell
## ## Refer to http://gking.harvard.edu/amelia/ for more information
## ##
```

```r
library(mice)
```

```
## 
## Attaching package: 'mice'
```

```
## The following object is masked from 'package:stats':
## 
##     filter
```

```
## The following objects are masked from 'package:base':
## 
##     cbind, rbind
```

```r
library(GGally)
```

```
## Registered S3 method overwritten by 'GGally':
##   method from   
##   +.gg   ggplot2
```

```r
library(rpart)
```

# 3. DATA SOURCING


```r
commerce <- fread('http://bit.ly/EcommerceCustomersDataset')
```

```
## Warning in download.file(input, tmpFile, method = method, mode = "wb", quiet = !
## showProgress): downloaded length 700416 != reported length 1071975
```

```
## Warning in fread("http://bit.ly/
## EcommerceCustomersDataset"): Discarded single-line footer:
## <<1,20,0,0,21,442,0.008695652,0.020289855,42.28127491,0,Nov,2,2,1,13,Retu>>
```

```r
# View the dataset in our environment
View(commerce)
```

# 4. PREVIEWING THE DATASET

i)  The top 6 rows in our dataset


```r
head(commerce)
```

```
##    Administrative Administrative_Duration Informational Informational_Duration
## 1:              0                       0             0                      0
## 2:              0                       0             0                      0
## 3:              0                      -1             0                     -1
## 4:              0                       0             0                      0
## 5:              0                       0             0                      0
## 6:              0                       0             0                      0
##    ProductRelated ProductRelated_Duration BounceRates ExitRates PageValues
## 1:              1                0.000000  0.20000000 0.2000000          0
## 2:              2               64.000000  0.00000000 0.1000000          0
## 3:              1               -1.000000  0.20000000 0.2000000          0
## 4:              2                2.666667  0.05000000 0.1400000          0
## 5:             10              627.500000  0.02000000 0.0500000          0
## 6:             19              154.216667  0.01578947 0.0245614          0
##    SpecialDay Month OperatingSystems Browser Region TrafficType
## 1:          0   Feb                1       1      1           1
## 2:          0   Feb                2       2      1           2
## 3:          0   Feb                4       1      9           3
## 4:          0   Feb                3       2      2           4
## 5:          0   Feb                3       3      1           4
## 6:          0   Feb                2       2      1           3
##          VisitorType Weekend Revenue
## 1: Returning_Visitor   FALSE   FALSE
## 2: Returning_Visitor   FALSE   FALSE
## 3: Returning_Visitor   FALSE   FALSE
## 4: Returning_Visitor   FALSE   FALSE
## 5: Returning_Visitor    TRUE   FALSE
## 6: Returning_Visitor   FALSE   FALSE
```

ii) The bottom 6 rows in our dataset


```r
tail(commerce)
```

```
##    Administrative Administrative_Duration Informational Informational_Duration
## 1:              0                       0             0                      0
## 2:              2                      28             0                      0
## 3:              0                       0             0                      0
## 4:              0                       0             0                      0
## 5:              5                     150             0                      0
## 6:              6                     416             2                    121
##    ProductRelated ProductRelated_Duration BounceRates   ExitRates PageValues
## 1:             11                 499.000 0.000000000 0.018181818          0
## 2:             27                 434.500 0.000000000 0.006896552          0
## 3:             23                1375.917 0.039130435 0.093478261          0
## 4:              7                  87.000 0.000000000 0.028571429          0
## 5:            135                4966.521 0.003649635 0.011277952          0
## 6:             94                2881.685 0.006122449 0.025340136          0
##    SpecialDay Month OperatingSystems Browser Region TrafficType
## 1:          0   Dec                4       1      1           2
## 2:          0   Nov                8       2      7           2
## 3:          0   Nov                3       2      8          13
## 4:          0   Dec                8      13      9          20
## 5:          0   Nov                1       1      2          10
## 6:          0   Nov                3       2      2          13
##          VisitorType Weekend Revenue
## 1: Returning_Visitor   FALSE   FALSE
## 2:       New_Visitor   FALSE    TRUE
## 3: Returning_Visitor   FALSE   FALSE
## 4:             Other   FALSE   FALSE
## 5: Returning_Visitor    TRUE   FALSE
## 6: Returning_Visitor   FALSE   FALSE
```

iii) The shape of the dataset


```r
dim(commerce)
```

```
## [1] 8118   18
```

Our dataset has 12330 rows and 18 columns. 

iv) The datatypes of the columns in our dataset


```r
str(commerce)
```

```
## Classes 'data.table' and 'data.frame':	8118 obs. of  18 variables:
##  $ Administrative         : int  0 0 0 0 0 0 0 1 0 0 ...
##  $ Administrative_Duration: num  0 0 -1 0 0 0 -1 -1 0 0 ...
##  $ Informational          : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ Informational_Duration : num  0 0 -1 0 0 0 -1 -1 0 0 ...
##  $ ProductRelated         : int  1 2 1 2 10 19 1 1 2 3 ...
##  $ ProductRelated_Duration: num  0 64 -1 2.67 627.5 ...
##  $ BounceRates            : num  0.2 0 0.2 0.05 0.02 ...
##  $ ExitRates              : num  0.2 0.1 0.2 0.14 0.05 ...
##  $ PageValues             : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ SpecialDay             : num  0 0 0 0 0 0 0.4 0 0.8 0.4 ...
##  $ Month                  : chr  "Feb" "Feb" "Feb" "Feb" ...
##  $ OperatingSystems       : int  1 2 4 3 3 2 2 1 2 2 ...
##  $ Browser                : int  1 2 1 2 3 2 4 2 2 4 ...
##  $ Region                 : int  1 1 9 2 1 1 3 1 2 1 ...
##  $ TrafficType            : int  1 2 3 4 4 3 3 5 3 2 ...
##  $ VisitorType            : chr  "Returning_Visitor" "Returning_Visitor" "Returning_Visitor" "Returning_Visitor" ...
##  $ Weekend                : logi  FALSE FALSE FALSE FALSE TRUE FALSE ...
##  $ Revenue                : logi  FALSE FALSE FALSE FALSE FALSE FALSE ...
##  - attr(*, ".internal.selfref")=<externalptr>
```

We have 2 logical columns, 7 numeric columns, 7 integer columns and 2 columns of the datatype character. 

- Before cleaning our dataset, we can go ahead and convert the datatypes of some of our numerical columns and make them categorical for better analysis. 


```r
commerce$OperatingSystems <- as.character(commerce$OperatingSystems)
commerce$Browser <- as.character(commerce$Browser)
commerce$Region <- as.character(commerce$Region)
commerce$TrafficType <- as.character(commerce$TrafficType)
```

# 5. CLEANING THE DATASET

## Checking for null values in the dataset


```r
colSums(is.na(commerce))
```

```
##          Administrative Administrative_Duration           Informational 
##                      14                      14                      14 
##  Informational_Duration          ProductRelated ProductRelated_Duration 
##                      14                      14                      14 
##             BounceRates               ExitRates              PageValues 
##                      14                      14                       0 
##              SpecialDay                   Month        OperatingSystems 
##                       0                       0                       0 
##                 Browser                  Region             TrafficType 
##                       0                       0                       0 
##             VisitorType                 Weekend                 Revenue 
##                       0                       0                       0
```

From the code above, we can tell that we have 14 missing values in each of the following 8 columns namely : "Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates" and "ExitRates".

### Dealing with the missing values in our dataset


```r
com <- na.omit(commerce)
dim(commerce)
```

```
## [1] 8118   18
```

```r
dim(com)
```

```
## [1] 8104   18
```

We decided to omit the missing values from our initial dataset and use the new dataset for analysis and modelling. 

## Checking for duplicated rows in our dataset


```r
duplicated_rows <- com[(duplicated(com))]
duplicated_rows
```

```
##     Administrative Administrative_Duration Informational Informational_Duration
##  1:              0                       0             0                      0
##  2:              0                       0             0                      0
##  3:              0                       0             0                      0
##  4:              0                       0             0                      0
##  5:              0                       0             0                      0
##  6:              0                       0             0                      0
##  7:              0                       0             0                      0
##  8:              0                       0             0                      0
##  9:              0                       0             0                      0
## 10:              0                       0             0                      0
## 11:              0                       0             0                      0
## 12:              0                       0             0                      0
## 13:              0                       0             0                      0
## 14:              0                       0             0                      0
## 15:              0                       0             0                      0
## 16:              0                       0             0                      0
## 17:              0                       0             0                      0
## 18:              0                       0             0                      0
## 19:              0                       0             0                      0
## 20:              0                       0             0                      0
## 21:              0                       0             0                      0
## 22:              0                       0             0                      0
## 23:              0                       0             0                      0
## 24:              0                       0             0                      0
## 25:              0                       0             0                      0
## 26:              0                       0             0                      0
## 27:              0                       0             0                      0
## 28:              0                       0             0                      0
## 29:              0                       0             0                      0
## 30:              0                       0             0                      0
## 31:              0                       0             0                      0
## 32:              0                       0             0                      0
## 33:              0                       0             0                      0
## 34:              0                       0             0                      0
## 35:              0                       0             0                      0
## 36:              0                       0             0                      0
## 37:              0                       0             0                      0
## 38:              0                       0             0                      0
## 39:              0                       0             0                      0
## 40:              0                       0             0                      0
## 41:              0                       0             0                      0
## 42:              0                       0             0                      0
## 43:              0                       0             0                      0
## 44:              0                       0             0                      0
## 45:              0                       0             0                      0
## 46:              0                       0             0                      0
## 47:              0                       0             0                      0
## 48:              0                       0             0                      0
## 49:              0                       0             0                      0
## 50:              0                       0             0                      0
## 51:              0                       0             0                      0
## 52:              0                       0             0                      0
## 53:              0                       0             0                      0
## 54:              0                       0             0                      0
## 55:              0                       0             0                      0
## 56:              0                       0             0                      0
## 57:              0                       0             0                      0
## 58:              0                       0             0                      0
## 59:              0                       0             0                      0
## 60:              0                       0             0                      0
## 61:              0                       0             0                      0
## 62:              0                       0             0                      0
## 63:              0                       0             0                      0
## 64:              0                       0             0                      0
## 65:              0                       0             0                      0
## 66:              0                       0             0                      0
## 67:              0                       0             0                      0
## 68:              0                       0             0                      0
## 69:              0                       0             0                      0
## 70:              0                       0             0                      0
## 71:              0                       0             0                      0
## 72:              0                       0             0                      0
## 73:              0                       0             0                      0
## 74:              0                       0             0                      0
## 75:              0                       0             0                      0
## 76:              0                       0             0                      0
## 77:              0                       0             0                      0
## 78:              0                       0             0                      0
## 79:              0                       0             0                      0
## 80:              0                       0             0                      0
## 81:              0                       0             0                      0
##     Administrative Administrative_Duration Informational Informational_Duration
##     ProductRelated ProductRelated_Duration BounceRates ExitRates PageValues
##  1:              1                       0         0.2       0.2          0
##  2:              1                       0         0.2       0.2          0
##  3:              1                       0         0.2       0.2          0
##  4:              1                       0         0.2       0.2          0
##  5:              1                       0         0.2       0.2          0
##  6:              1                       0         0.2       0.2          0
##  7:              1                       0         0.2       0.2          0
##  8:              1                       0         0.2       0.2          0
##  9:              2                       0         0.2       0.2          0
## 10:              1                       0         0.2       0.2          0
## 11:              1                       0         0.2       0.2          0
## 12:              1                       0         0.2       0.2          0
## 13:              1                       0         0.2       0.2          0
## 14:              1                       0         0.2       0.2          0
## 15:              1                       0         0.2       0.2          0
## 16:              1                       0         0.2       0.2          0
## 17:              1                       0         0.2       0.2          0
## 18:              1                       0         0.2       0.2          0
## 19:              1                       0         0.2       0.2          0
## 20:              1                       0         0.2       0.2          0
## 21:              1                       0         0.2       0.2          0
## 22:              2                       0         0.2       0.2          0
## 23:              1                       0         0.2       0.2          0
## 24:              2                       0         0.2       0.2          0
## 25:              1                       0         0.2       0.2          0
## 26:              1                       0         0.2       0.2          0
## 27:              1                       0         0.2       0.2          0
## 28:              1                       0         0.2       0.2          0
## 29:              1                       0         0.2       0.2          0
## 30:              1                       0         0.2       0.2          0
## 31:              1                       0         0.2       0.2          0
## 32:              1                       0         0.2       0.2          0
## 33:              1                       0         0.2       0.2          0
## 34:              1                       0         0.2       0.2          0
## 35:              1                       0         0.2       0.2          0
## 36:              1                       0         0.2       0.2          0
## 37:              1                       0         0.2       0.2          0
## 38:              1                       0         0.2       0.2          0
## 39:              1                       0         0.2       0.2          0
## 40:              1                       0         0.2       0.2          0
## 41:              1                       0         0.2       0.2          0
## 42:              1                       0         0.2       0.2          0
## 43:              1                       0         0.2       0.2          0
## 44:              1                       0         0.2       0.2          0
## 45:              1                       0         0.2       0.2          0
## 46:              1                       0         0.2       0.2          0
## 47:              1                       0         0.2       0.2          0
## 48:              1                       0         0.2       0.2          0
## 49:              1                       0         0.2       0.2          0
## 50:              1                       0         0.2       0.2          0
## 51:              1                       0         0.2       0.2          0
## 52:              1                       0         0.2       0.2          0
## 53:              1                       0         0.2       0.2          0
## 54:              1                       0         0.2       0.2          0
## 55:              1                       0         0.2       0.2          0
## 56:              1                       0         0.2       0.2          0
## 57:              1                       0         0.2       0.2          0
## 58:              1                       0         0.2       0.2          0
## 59:              1                       0         0.2       0.2          0
## 60:              1                       0         0.2       0.2          0
## 61:              1                       0         0.2       0.2          0
## 62:              1                       0         0.2       0.2          0
## 63:              1                       0         0.2       0.2          0
## 64:              2                       0         0.2       0.2          0
## 65:              1                       0         0.2       0.2          0
## 66:              1                       0         0.2       0.2          0
## 67:              1                       0         0.2       0.2          0
## 68:              1                       0         0.2       0.2          0
## 69:              1                       0         0.2       0.2          0
## 70:              1                       0         0.2       0.2          0
## 71:              1                       0         0.2       0.2          0
## 72:              1                       0         0.2       0.2          0
## 73:              2                       0         0.2       0.2          0
## 74:              1                       0         0.2       0.2          0
## 75:              1                       0         0.2       0.2          0
## 76:              1                       0         0.2       0.2          0
## 77:              1                       0         0.2       0.2          0
## 78:              1                       0         0.2       0.2          0
## 79:              1                       0         0.2       0.2          0
## 80:              1                       0         0.2       0.2          0
## 81:              1                       0         0.2       0.2          0
##     ProductRelated ProductRelated_Duration BounceRates ExitRates PageValues
##     SpecialDay Month OperatingSystems Browser Region TrafficType
##  1:        0.0   Feb                1       1      1           3
##  2:        0.0   Feb                3       2      3           3
##  3:        0.0   Mar                1       1      1           1
##  4:        0.0   Mar                2       2      4           1
##  5:        0.0   Mar                3       2      3           1
##  6:        0.0   Mar                2       2      1           1
##  7:        0.0   Mar                2       2      1           1
##  8:        0.0   Mar                2       2      1           1
##  9:        0.0   Mar                2       5      1           1
## 10:        0.0   Mar                2       2      4           1
## 11:        0.0   Mar                3       2      3           1
## 12:        0.0   Mar                1       1      2           1
## 13:        0.0   Mar                3       2      2           1
## 14:        0.0   Mar                2       2      1           1
## 15:        0.0   Mar                2       2      1           1
## 16:        0.0   Mar                2       2      1           1
## 17:        0.0   Mar                2       2      1           1
## 18:        0.0   Mar                3       2      1           1
## 19:        0.0   Mar                2       4      1           1
## 20:        0.0   Mar                3       2      3           1
## 21:        0.0   Mar                1       1      1           3
## 22:        0.0   Mar                2       2      1           1
## 23:        0.0   Mar                1       1      3           3
## 24:        0.0   Mar                1       1      1           1
## 25:        0.0   Mar                1       1      8           1
## 26:        0.0   Mar                1       1      4           1
## 27:        0.0   Mar                2       2      1           1
## 28:        0.0   Mar                2       2      1           1
## 29:        0.0   Mar                3       2      3           1
## 30:        0.0   Mar                2       2      1           1
## 31:        0.0   Mar                1       1      1           3
## 32:        0.0   Mar                2       2      1           1
## 33:        0.0   Mar                2       2      7           1
## 34:        0.0   Mar                2       2      2           1
## 35:        0.0   Mar                3       2      1           1
## 36:        0.0   Mar                1       1      8           1
## 37:        0.0   Mar                2       2      1           3
## 38:        0.0   Mar                1       1      1           9
## 39:        0.0   Mar                3       2      1           1
## 40:        0.0   Mar                2       2      1           1
## 41:        0.0   Mar                2       2      1           1
## 42:        0.0   Mar                3       2      3           1
## 43:        0.0   Mar                2       4      1           1
## 44:        0.0   May                1       1      4           3
## 45:        0.0   May                1       1      1           3
## 46:        0.0   May                2       2      1           1
## 47:        0.0   May                2       4      1           3
## 48:        0.0   May                1       1      3           3
## 49:        0.0   May                1       1      1           3
## 50:        0.0   May                2       2      1           4
## 51:        0.0   May                2       2      4           1
## 52:        0.0   May                1       1      1           3
## 53:        0.0   May                1       1      4           3
## 54:        0.0   May                2       2      7           4
## 55:        0.0   May                1       1      4           3
## 56:        0.0   May                1       1      1           3
## 57:        0.0   May                2       2      2           1
## 58:        0.0   May                3       2      1          13
## 59:        0.0   May                2       2      1           3
## 60:        0.0   May                2       2      1           3
## 61:        0.0   May                2       2      1           3
## 62:        0.0   May                1       1      1           3
## 63:        0.0   May                3       2      9           3
## 64:        0.0   May                2       2      2           3
## 65:        0.0   May                2       2      1           3
## 66:        0.0   May                2       2      1           3
## 67:        0.8   May                2       2      1           1
## 68:        0.0   May                3       2      3           3
## 69:        0.0   May                2       2      1           3
## 70:        0.0   May                2       2      6           3
## 71:        0.0   May                1       1      6           4
## 72:        0.0   May                2       2      1          13
## 73:        0.0   May                2       2      2           3
## 74:        0.6   May                2       2      1           1
## 75:        0.0   May                3       2      3          13
## 76:        0.0   May                1       1      3          15
## 77:        0.0   May                1       1      3           3
## 78:        0.0   May                2       4      1           6
## 79:        0.0  June                2       2      1           1
## 80:        0.0  June                2       2      1           1
## 81:        0.0  June                3       2      3          13
##     SpecialDay Month OperatingSystems Browser Region TrafficType
##           VisitorType Weekend Revenue
##  1: Returning_Visitor   FALSE   FALSE
##  2: Returning_Visitor   FALSE   FALSE
##  3: Returning_Visitor    TRUE   FALSE
##  4: Returning_Visitor   FALSE   FALSE
##  5: Returning_Visitor   FALSE   FALSE
##  6: Returning_Visitor   FALSE   FALSE
##  7: Returning_Visitor   FALSE   FALSE
##  8: Returning_Visitor   FALSE   FALSE
##  9: Returning_Visitor   FALSE   FALSE
## 10: Returning_Visitor   FALSE   FALSE
## 11: Returning_Visitor   FALSE   FALSE
## 12: Returning_Visitor   FALSE   FALSE
## 13: Returning_Visitor   FALSE   FALSE
## 14: Returning_Visitor   FALSE   FALSE
## 15: Returning_Visitor   FALSE   FALSE
## 16: Returning_Visitor   FALSE   FALSE
## 17: Returning_Visitor   FALSE   FALSE
## 18: Returning_Visitor   FALSE   FALSE
## 19: Returning_Visitor   FALSE   FALSE
## 20: Returning_Visitor   FALSE   FALSE
## 21: Returning_Visitor   FALSE   FALSE
## 22: Returning_Visitor   FALSE   FALSE
## 23: Returning_Visitor   FALSE   FALSE
## 24: Returning_Visitor   FALSE   FALSE
## 25: Returning_Visitor   FALSE   FALSE
## 26: Returning_Visitor   FALSE   FALSE
## 27: Returning_Visitor   FALSE   FALSE
## 28: Returning_Visitor   FALSE   FALSE
## 29: Returning_Visitor   FALSE   FALSE
## 30: Returning_Visitor   FALSE   FALSE
## 31: Returning_Visitor    TRUE   FALSE
## 32: Returning_Visitor   FALSE   FALSE
## 33: Returning_Visitor   FALSE   FALSE
## 34: Returning_Visitor   FALSE   FALSE
## 35: Returning_Visitor   FALSE   FALSE
## 36: Returning_Visitor   FALSE   FALSE
## 37: Returning_Visitor   FALSE   FALSE
## 38: Returning_Visitor    TRUE   FALSE
## 39: Returning_Visitor   FALSE   FALSE
## 40: Returning_Visitor   FALSE   FALSE
## 41: Returning_Visitor   FALSE   FALSE
## 42: Returning_Visitor   FALSE   FALSE
## 43: Returning_Visitor   FALSE   FALSE
## 44: Returning_Visitor   FALSE   FALSE
## 45: Returning_Visitor   FALSE   FALSE
## 46: Returning_Visitor   FALSE   FALSE
## 47: Returning_Visitor   FALSE   FALSE
## 48: Returning_Visitor   FALSE   FALSE
## 49: Returning_Visitor   FALSE   FALSE
## 50: Returning_Visitor   FALSE   FALSE
## 51: Returning_Visitor   FALSE   FALSE
## 52: Returning_Visitor   FALSE   FALSE
## 53: Returning_Visitor   FALSE   FALSE
## 54: Returning_Visitor   FALSE   FALSE
## 55: Returning_Visitor   FALSE   FALSE
## 56: Returning_Visitor   FALSE   FALSE
## 57: Returning_Visitor   FALSE   FALSE
## 58: Returning_Visitor   FALSE   FALSE
## 59: Returning_Visitor   FALSE   FALSE
## 60: Returning_Visitor   FALSE   FALSE
## 61: Returning_Visitor   FALSE   FALSE
## 62: Returning_Visitor   FALSE   FALSE
## 63: Returning_Visitor   FALSE   FALSE
## 64: Returning_Visitor   FALSE   FALSE
## 65: Returning_Visitor   FALSE   FALSE
## 66: Returning_Visitor   FALSE   FALSE
## 67: Returning_Visitor   FALSE   FALSE
## 68: Returning_Visitor   FALSE   FALSE
## 69: Returning_Visitor   FALSE   FALSE
## 70: Returning_Visitor   FALSE   FALSE
## 71: Returning_Visitor    TRUE   FALSE
## 72: Returning_Visitor   FALSE   FALSE
## 73: Returning_Visitor   FALSE   FALSE
## 74: Returning_Visitor   FALSE   FALSE
## 75: Returning_Visitor   FALSE   FALSE
## 76: Returning_Visitor   FALSE   FALSE
## 77: Returning_Visitor   FALSE   FALSE
## 78: Returning_Visitor   FALSE   FALSE
## 79: Returning_Visitor   FALSE   FALSE
## 80: Returning_Visitor   FALSE   FALSE
## 81: Returning_Visitor   FALSE   FALSE
##           VisitorType Weekend Revenue
```

- We can tell that we have 117 duplicated rows which we will go ahead and drop them since they will distort our analysis. 

### Dropping the duplicated rows


```r
# We create a new dataset that holds the unique values in our dataset
new_com <- unique(com)
dim(new_com)
```

```
## [1] 8023   18
```

- After dropping the duplicated rows, we go ahead to use the new_com dataset for analysis. 

## Checking for outliers


```r
par(mfrow = c(4,3), mar = c(5,4,3,3))

# Finding all columns that are numerical/not strings & subsetting to new dataframe
#numerical_col <- new_com[, sapply(new_com, is.numeric)]

numerical_col <- subset(new_com, select = c(1,2,3,4,5,6,7,8,9,10))
num1 <- subset(new_com, select = c(1,2,3))
num2 <- subset(new_com, select = c(4,5,6))
num3 <- subset(new_com, select = c(7,8,9))
#boxplot(numerical_col, main='BoxPlots')
```


```r
boxplot(num1)
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

- It is evident we have many outliers in the Administrative_Duration column which we can leave as they constitute majority of the column.


```r
boxplot(num2)
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

- We can also tell that the ProductRelated_Duration has so many outliers which we will also not drop as they constitute majority of the column. 


```r
boxplot(num3)
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

- The PageValues column also has so many outliers which we will not drop for the same reason given above. 


# 6. EXPLORATORY DATA ANALYSIS

## A. UNIVARIATE DATA ANALYSIS

### Measures of Central Tendency

#### i) Mean 


```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:data.table':
## 
##     between, first, last
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
new_com %>% summarise_if(is.numeric, mean)
```

```
##   Administrative Administrative_Duration Informational Informational_Duration
## 1       2.305621                80.89066     0.4588059               31.27817
##   ProductRelated ProductRelated_Duration BounceRates ExitRates PageValues
## 1       27.46005                1019.507   0.0213451 0.0427522   5.570478
##   SpecialDay
## 1 0.09422909
```

#### ii) Mode


```r
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

new_com %>% summarise_if(is.numeric, getmode)
```

```
##   Administrative Administrative_Duration Informational Informational_Duration
## 1              0                       0             0                      0
##   ProductRelated ProductRelated_Duration BounceRates ExitRates PageValues
## 1              1                       0           0       0.2          0
##   SpecialDay
## 1          0
```

#### iii) Median


```r
new_com %>% summarise_if(is.numeric, median)
```

```
##   Administrative Administrative_Duration Informational Informational_Duration
## 1              1                11.42857             0                      0
##   ProductRelated ProductRelated_Duration BounceRates ExitRates PageValues
## 1             17                   533.5 0.002272727     0.026          0
##   SpecialDay
## 1          0
```


### Measures of Dispersion

#### i) Range


```r
new_com %>% summarise_if(is.numeric, range)
```

```
##   Administrative Administrative_Duration Informational Informational_Duration
## 1              0                   -1.00             0                 -1.000
## 2             24                 3398.75            24               2549.375
##   ProductRelated ProductRelated_Duration BounceRates ExitRates PageValues
## 1              0                   -1.00         0.0       0.0     0.0000
## 2            705                63973.52         0.2       0.2   261.4913
##   SpecialDay
## 1          0
## 2          1
```

#### ii) Quantiles


```r
new_com %>% summarise_if(is.numeric, quantile)
```

```
##   Administrative Administrative_Duration Informational Informational_Duration
## 1              0                -1.00000             0                 -1.000
## 2              0                 0.00000             0                  0.000
## 3              1                11.42857             0                  0.000
## 4              4                94.00000             0                  0.000
## 5             24              3398.75000            24               2549.375
##   ProductRelated ProductRelated_Duration BounceRates  ExitRates PageValues
## 1              0                 -1.0000 0.000000000 0.00000000     0.0000
## 2              7                177.8167 0.000000000 0.01337456     0.0000
## 3             17                533.5000 0.002272727 0.02600000     0.0000
## 4             33               1264.5417 0.018181818 0.05000000     0.0000
## 5            705              63973.5222 0.200000000 0.20000000   261.4913
##   SpecialDay
## 1          0
## 2          0
## 3          0
## 4          0
## 5          1
```

#### iii) Variance


```r
new_com %>% summarise_if(is.numeric, var)
```

```
##   Administrative Administrative_Duration Informational Informational_Duration
## 1       10.46679                30598.94      1.457758               17265.64
##   ProductRelated ProductRelated_Duration BounceRates   ExitRates PageValues
## 1       1391.922                 2937931  0.00212342 0.002225422   290.7173
##   SpecialDay
## 1 0.05760693
```

#### iv) Standard Deviation


```r
new_com %>% summarise_if(is.numeric, sd)
```

```
##   Administrative Administrative_Duration Informational Informational_Duration
## 1       3.235242                174.9255      1.207377               131.3988
##   ProductRelated ProductRelated_Duration BounceRates  ExitRates PageValues
## 1       37.30848                1714.039  0.04608058 0.04717438   17.05044
##   SpecialDay
## 1  0.2400144
```

### Frequency Tables


```r
month <- table(new_com$Month)
month
```

```
## 
##  Aug  Dec  Feb  Jul June  Mar  May  Nov  Oct  Sep 
##  433   57  182  432  285 1853 3328  456  549  448
```

May had the most entries in our dataset followed by November then March.


```r
os <- table(new_com$OperatingSystems)
os
```

```
## 
##    1    2    3    4    5    6    7    8 
## 1616 4415 1651  312    3   14    6    6
```

Operating system 2 had the most entries in our dataset. 


```r
browser <- table(new_com$Browser)
browser
```

```
## 
##    1   10   11   12   13    2    3    4    5    6    7    8    9 
## 1568  110    3    2    4 5125   79  545  332  136   37   81    1
```

Browser 2 had the most entries in our dataset. 

```r
region <- table(new_com$Region)
region
```

```
## 
##    1    2    3    4    5    6    7    8    9 
## 2988  802 1558  784  232  585  510  298  266
```

Region 1 had the most entries in our dataset. 


```r
traffic <- table(new_com$TrafficType)
traffic
```

```
## 
##    1   10   11   12   13   14   15   16   17   18   19    2   20    3    4    5 
## 1632  144  117    1  513   13   28    3    1   10   17 2097  101 1521 1066  260 
##    6    7    8    9 
##  344   24   90   41
```

Traffic Type 2 had the most entries in our dataset. 


```r
visitor <- table(new_com$VisitorType)
visitor
```

```
## 
##       New_Visitor             Other Returning_Visitor 
##              1050                 4              6969
```


```r
weekend <- table(new_com$Weekend)
weekend
```

```
## 
## FALSE  TRUE 
##  6212  1811
```


```r
revenue <- table(new_com$Revenue)
revenue
```

```
## 
## FALSE  TRUE 
##  6976  1047
```

### Graphical Plots

#### i) Bar Charts

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-30-1.png)<!-- -->

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-31-1.png)<!-- -->

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-32-1.png)<!-- -->

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-33-1.png)<!-- -->


## 2. BIVARIATE DATA ANALYSIS

- First we can create variables that will hold the numerical columns in our dataset. 


```r
admin <- new_com$Administrative
addur <- new_com$Administrative_Duration
info <- new_com$Informational
infdur <- new_com$Informational_Duration
prod <- new_com$ProductRelated
prdur <- new_com$ProductRelated_Duration
bounce <- new_com$BounceRates
exit <- new_com$ExitRates
page <- new_com$PageValues
special <- new_com$SpecialDay
```

### Covariance


```r
cov(admin, info)
```

```
## [1] 1.442285
```

There is positive covariance between the administrative and informational columns. 


```r
cov(info, prod)
```

```
## [1] 15.13769
```

There is positive covariance between the informational and product related columns. 


```r
cov(bounce, exit)
```

```
## [1] 0.001959258
```

- There is a positive covariance between the bounce rates and exit rates columns. 


```r
cov(page, special)
```

```
## [1] -0.3296325
```

- There is a negative covariance between the page values and the special day columns. 


### Correlation 

- The corrplot library allows us to plot a correlation matrix of the numerical columns in our dataset. 


```r
library(corrplot)
```

```
## corrplot 0.90 loaded
```

- Plot the correlation map. 


```r
# Correlation matrix
correlation <- cor(numerical_col)
corrplot(correlation)
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-40-1.png)<!-- -->

### Scatter Plots 


```r
# First we import the ggplot2 library which will help us in visualizations
library(ggplot2)
```


```r
ggplot(new_com, aes(admin, info)) + geom_point()
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-42-1.png)<!-- -->


```r
ggplot(new_com, aes(prod, info)) + geom_point()
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-43-1.png)<!-- -->


# 7. IMPLEMENTING THE SOLUTION

Perform clustering and upon implementation, provide comparisons between the approaches learned this week i.e. K-Means clustering vs Hierarchical clustering highlighting the strengths and limitations of each approach in the context of your analysis. 

## 1. K - Means Clustering

- First normalize the data. 


```r
# normalizing the data
normalize <- function(x){
  return ((x-min(x)) / (max(x)-min(x)))
}

com.norm <- normalize(numerical_col)
```

- Sample the data to allow clustering since our initial dataset is quite huge. 


```r
# this step was added because finding the optimum number of clusters took a while to run. Therefore we take a random sample of 1300 which is around 10% of the initial dataset
Com <- com.norm[sample(nrow(com.norm), size=1300), ]
head(Com)
```

```
##    Administrative Administrative_Duration Informational Informational_Duration
## 1:   1.563122e-05            1.563122e-05  1.563122e-05           1.563122e-05
## 2:   1.406810e-04            1.980997e-03  1.563122e-05           1.563122e-05
## 3:   1.563122e-05            1.563122e-05  1.563122e-05           1.563122e-05
## 4:   7.815611e-05            6.950684e-04  1.563122e-05           1.563122e-05
## 5:   6.252489e-05            7.096575e-03  6.252489e-05           7.729640e-03
## 6:   1.563122e-05            1.563122e-05  1.563122e-05           1.563122e-05
##    ProductRelated ProductRelated_Duration  BounceRates    ExitRates
## 1:   3.126245e-05            1.563122e-05 1.875747e-05 1.875747e-05
## 2:   1.250498e-04            1.292181e-03 1.615226e-05 1.600898e-05
## 3:   5.470928e-04            1.226374e-02 1.563122e-05 1.572317e-05
## 4:   7.815611e-05            4.418426e-04 1.563122e-05 1.607783e-05
## 5:   5.783552e-04            1.876564e-02 1.570566e-05 1.581152e-05
## 6:   9.378734e-05            3.907806e-04 1.563122e-05 1.641278e-05
##      PageValues   SpecialDay
## 1: 1.563122e-05 1.563122e-05
## 2: 1.563122e-05 1.563122e-05
## 3: 9.859991e-04 1.563122e-05
## 4: 1.563122e-05 1.563122e-05
## 5: 1.613694e-04 1.563122e-05
## 6: 1.563122e-05 1.875747e-05
```

- Load the factoextra library which holds the clustering techniques. 


```r
# Loading the required libraries
library(factoextra)
```

```
## Welcome! Want to learn more? See two factoextra-related books at https://goo.gl/ve3WBa
```

```r
library(NbClust)
library (cluster)
```

- Determine the optimal number of k clusters using the elbow method. 


```r
# Elbow method
fviz_nbclust(Com, kmeans, method = "wss") +
            geom_vline(xintercept = 3, linetype = 2)+
            labs(subtitle = "Elbow method")
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-47-1.png)<!-- -->

- Using the elbow method, we were able to determine that the optimal number of k in our case is 3. 

- We can also dtermine the optimal number of k using the silhouette method and compare with the elbow method.


```r
# Silhouette method
fviz_nbclust(Com, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-48-1.png)<!-- -->

- Using the silhouette method, we were able to determine that the optimal number of k in our case is 2.

- We can also try using the gap statistic method with nboot set at 500.


```r
# Gap statistic
# nboot = 50 to keep the function speedy. 

set.seed(123)
fviz_nbclust(Com, kmeans, nstart = 25,  method = "gap_stat", nboot = 500)+
  labs(subtitle = "Gap statistic method")
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-49-1.png)<!-- -->

- Using the gap statistic method, we find the optimal number of clusters as 1. 


```r
# choosing the best number of clusters
nb<-NbClust(data = Com, distance = "euclidean",
        min.nc = 2, max.nc = 15, method = "kmeans")
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-50-1.png)<!-- -->

```
## *** : The Hubert index is a graphical method of determining the number of clusters.
##                 In the plot of Hubert index, we seek a significant knee that corresponds to a 
##                 significant increase of the value of the measure i.e the significant peak in Hubert
##                 index second differences plot. 
## 
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-50-2.png)<!-- -->

```
## *** : The D index is a graphical method of determining the number of clusters. 
##                 In the plot of D index, we seek a significant knee (the significant peak in Dindex
##                 second differences plot) that corresponds to a significant increase of the value of
##                 the measure. 
##  
## ******************************************************************* 
## * Among all indices:                                                
## * 5 proposed 2 as the best number of clusters 
## * 6 proposed 3 as the best number of clusters 
## * 3 proposed 4 as the best number of clusters 
## * 3 proposed 5 as the best number of clusters 
## * 2 proposed 12 as the best number of clusters 
## * 1 proposed 13 as the best number of clusters 
## * 1 proposed 14 as the best number of clusters 
## * 3 proposed 15 as the best number of clusters 
## 
##                    ***** Conclusion *****                            
##  
## * According to the majority rule, the best number of clusters is  3 
##  
##  
## *******************************************************************
```

```r
fviz_nbclust(nb)
```

```
## Warning in if (class(best_nc) == "numeric") print(best_nc) else if
## (class(best_nc) == : the condition has length > 1 and only the first element
## will be used
```

```
## Warning in if (class(best_nc) == "matrix") .viz_NbClust(x, print.summary, : the
## condition has length > 1 and only the first element will be used
```

```
## Warning in if (class(best_nc) == "numeric") print(best_nc) else if
## (class(best_nc) == : the condition has length > 1 and only the first element
## will be used
```

```
## Warning in if (class(best_nc) == "matrix") {: the condition has length > 1 and
## only the first element will be used
```

```
## Among all indices: 
## ===================
## * 2 proposed  0 as the best number of clusters
## * 5 proposed  2 as the best number of clusters
## * 6 proposed  3 as the best number of clusters
## * 3 proposed  4 as the best number of clusters
## * 3 proposed  5 as the best number of clusters
## * 2 proposed  12 as the best number of clusters
## * 1 proposed  13 as the best number of clusters
## * 1 proposed  14 as the best number of clusters
## * 3 proposed  15 as the best number of clusters
## 
## Conclusion
## =========================
## * According to the majority rule, the best number of clusters is  3 .
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-50-3.png)<!-- -->

- Fit the kmeans model with our data 


```r
km <- kmeans(Com,3,iter.max = 10, nstart = 25)
km
```

```
## K-means clustering with 3 clusters of sizes 89, 1208, 3
## 
## Cluster means:
##   Administrative Administrative_Duration Informational Informational_Duration
## 1   9.712434e-05             0.002631571  0.0000365314           0.0015080216
## 2   4.918400e-05             0.001138517  0.0000212859           0.0004035953
## 3   2.605204e-04             0.021273626  0.0002136267           0.0135599120
##   ProductRelated ProductRelated_Duration  BounceRates    ExitRates   PageValues
## 1   0.0017194345              0.08112895 1.574354e-05 1.598621e-05 7.251401e-05
## 2   0.0003442492              0.01096647 1.599926e-05 1.633493e-05 9.391137e-05
## 3   0.0094204168              0.47104755 1.575942e-05 1.597128e-05 2.578233e-05
##     SpecialDay
## 1 1.724703e-05
## 2 1.696402e-05
## 3 1.563122e-05
## 
## Clustering vector:
##    [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 1 2 3 2 2 2 2 2 2 1 2 2 2 2 2 2 2
##   [38] 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##   [75] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [112] 2 2 1 1 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2
##  [149] 2 1 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2
##  [186] 2 2 2 2 1 2 2 2 2 2 2 2 1 2 1 2 2 2 2 2 2 2 2 2 1 2 2 2 1 2 2 2 2 2 2 2 1
##  [223] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2
##  [260] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2
##  [297] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [334] 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [371] 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 1
##  [408] 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2
##  [445] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [482] 2 2 2 1 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [519] 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [556] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 1 2 1 1 1 2 2 2
##  [593] 1 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 1 2 1 2 2 2
##  [630] 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2
##  [667] 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 1 2
##  [704] 2 2 2 2 2 2 2 1 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 1 2 2 2 2 2 1
##  [741] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [778] 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [815] 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [852] 2 2 2 2 2 2 2 2 2 2 2 3 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
##  [889] 1 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 1 2 2 2 2 2 2 2 1 2 1 2 2 2
##  [926] 1 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 1 2
##  [963] 1 2 2 1 2 2 2 2 2 2 2 1 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2
## [1000] 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
## [1037] 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
## [1074] 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
## [1111] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
## [1148] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2
## [1185] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2
## [1222] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 1 2 2 2 2 2
## [1259] 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 1 2 2 2 2 2 2 2 2 2
## [1296] 2 2 2 1 2
## 
## Within cluster sum of squares by cluster:
## [1] 0.09348085 0.14855724 0.06358302
##  (between_SS / total_SS =  77.1 %)
## 
## Available components:
## 
## [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
## [6] "betweenss"    "size"         "iter"         "ifault"
```


```r
#plot results of final k-means model
fviz_cluster(km, data = Com)
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-52-1.png)<!-- -->


### Hierarchical Clustering

- For hierarchical, we first install and load the dplyr package. 


```r
# Installing the package
install.packages("dplyr")
```

```
## Warning: package 'dplyr' is in use and will not be installed
```

```r
# Loading package
library(dplyr)
	
# Summary of dataset in package
head(Com)
```

```
##    Administrative Administrative_Duration Informational Informational_Duration
## 1:   1.563122e-05            1.563122e-05  1.563122e-05           1.563122e-05
## 2:   1.406810e-04            1.980997e-03  1.563122e-05           1.563122e-05
## 3:   1.563122e-05            1.563122e-05  1.563122e-05           1.563122e-05
## 4:   7.815611e-05            6.950684e-04  1.563122e-05           1.563122e-05
## 5:   6.252489e-05            7.096575e-03  6.252489e-05           7.729640e-03
## 6:   1.563122e-05            1.563122e-05  1.563122e-05           1.563122e-05
##    ProductRelated ProductRelated_Duration  BounceRates    ExitRates
## 1:   3.126245e-05            1.563122e-05 1.875747e-05 1.875747e-05
## 2:   1.250498e-04            1.292181e-03 1.615226e-05 1.600898e-05
## 3:   5.470928e-04            1.226374e-02 1.563122e-05 1.572317e-05
## 4:   7.815611e-05            4.418426e-04 1.563122e-05 1.607783e-05
## 5:   5.783552e-04            1.876564e-02 1.570566e-05 1.581152e-05
## 6:   9.378734e-05            3.907806e-04 1.563122e-05 1.641278e-05
##      PageValues   SpecialDay
## 1: 1.563122e-05 1.563122e-05
## 2: 1.563122e-05 1.563122e-05
## 3: 9.859991e-04 1.563122e-05
## 4: 1.563122e-05 1.563122e-05
## 5: 1.613694e-04 1.563122e-05
## 6: 1.563122e-05 1.875747e-05
```

- Find the euclidean distance matrix in our dataset. 


```r
# Finding distance matrix
distance <- dist(Com, method = 'euclidean')
```

- Fit the hierarchical model to our train dataset. 


```r
# Fitting Hierarchical clustering Model
# to training dataset
set.seed(240) # Setting seed
Hierar_cl <- hclust(distance, method = "average")
```

- Plot the dendogram of our model. 


```r
# Plotting dendrogram
plot(Hierar_cl)
```

![](Week-13-Ecommerce-Project_files/figure-html/unnamed-chunk-56-1.png)<!-- -->

- Cut the tree by height 


```r
# Choosing no. of clusters
# Cutting tree by height
#abline(h = 110, col = "green")
```

- Cutting the tree by the number of clusters. 


```r
# Cutting tree by no. of clusters
fit <- cutree(Hierar_cl, k = 3 )
fit
```

```
##    [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##   [38] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##   [75] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [112] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [149] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [186] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [223] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [260] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [297] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [334] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [371] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [408] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [445] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [482] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [519] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [556] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [593] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [630] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [667] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [704] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [741] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [778] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [815] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [852] 1 1 1 1 1 1 1 1 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [889] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [926] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
##  [963] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1000] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1037] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1074] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1111] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1148] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1185] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1222] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1259] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
## [1296] 1 1 1 1 1
```


```r
table(fit)
```

```
## fit
##    1    2    3 
## 1297    1    2
```

```r
#rect.hclust(Hierar_cl, k = 3, border = "green")
```

# 8. CHALLENGING THE SOLUTION 


```r
# We could challenge the solution by using different numbers of k clusters and compare the results of the algorithms. 
# Using the elbow method, we were able to determine that 3 was the optimal number of k clusters in our dataset. 
# Using the silhouette method, 2 was the optimal number of k clusters 
# Using the gap statistic method, 1 was the optimal number of k clusters. 
```

# 9. FOLLOW UP QUESTIONS

## i) Did we have the right data ?

Yes we did as we were also given the variable definitions which helped us understand the dataset better. 






























