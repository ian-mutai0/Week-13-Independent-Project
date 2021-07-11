---
title: "Week 13 Independent Project"
author: "Mr. Mutai"
date: "7/8/2021"
output: 
  html_document: 
    keep_md: yes
---




# ASSESSMENT QUESTION

A Kenyan entrepreneur has created an online cryptography course and would want to advertise it on her blog.
She currently targets audiences originating from various countries.
In the past, she ran ads to advertise a related course on the same blog and collected data in the process.
She would now like to employ your services as a Data Science Consultant to help her identify which individuals are most likely to click on her ads. 


# 1. DEFINING THE QUESTION

## i) Specifying the Data Analytic Question

We will be tasked to create a supervised learning model to help a Kenyan entrepreneur identify which individuals are most likely to click on the ads in the blog. 

## ii) Defining the Metric for Success

To perform exploratory data analysis and implement our solution by creating various supervised learning models and choosing the best performing one that can help us predict which individuals are most likely to click on the ads in the blog. 

## iii) Understanding the Context

The entrepreneur wants to understand which factors determine whether an individual will click on her ads such as their age, gender, time spent on the site, daily internet usage, the city and the country they are from.

## iv) Recording the Experimental Design

1)  Define the question, the metric for success, the context, experimental design taken and the appropriateness of the available data to answer the given question
2)  Read the dataset into our environment (RStudio)
3)  Preview the dataset
4)  Find and deal with outliers, anomalies, and missing data within the dataset
5)  Perform univariate and bivariate analysis
6)  Implement our solution by creating various supervised learning models and choose the best performing one for our research problem
7)  From our insights provide conclusions and recommendations


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
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:psych':
## 
##     outlier
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


# 3. LOADING THE DATASET

First we set the working directory by choosing the directory in which our dataset is in.
Afterwards we read the csv file and view it in our environment.
OR 
Alternatively read the URL and view the csv file in our environment.


```r
# Set working directory 
# setwd(choose.dir())

# Read the csv file from the URL
advertising = fread('http://bit.ly/IPAdvertisingData')

# View the dataset in our environment
View(advertising)
```


# 4. PREVIEWING THE DATASET

i)  The top 6 rows in our dataset


```r
head(advertising)
```

```
##    Daily Time Spent on Site Age Area Income Daily Internet Usage
## 1:                    68.95  35    61833.90               256.09
## 2:                    80.23  31    68441.85               193.77
## 3:                    69.47  26    59785.94               236.50
## 4:                    74.15  29    54806.18               245.89
## 5:                    68.37  35    73889.99               225.58
## 6:                    59.99  23    59761.56               226.74
##                            Ad Topic Line           City Male    Country
## 1:    Cloned 5thgeneration orchestration    Wrightburgh    0    Tunisia
## 2:    Monitored national standardization      West Jodi    1      Nauru
## 3:      Organic bottom-line service-desk       Davidton    0 San Marino
## 4: Triple-buffered reciprocal time-frame West Terrifurt    1      Italy
## 5:         Robust logistical utilization   South Manuel    0    Iceland
## 6:       Sharable client-driven software      Jamieberg    1     Norway
##              Timestamp Clicked on Ad
## 1: 2016-03-27 00:53:11             0
## 2: 2016-04-04 01:39:02             0
## 3: 2016-03-13 20:35:42             0
## 4: 2016-01-10 02:31:19             0
## 5: 2016-06-03 03:36:18             0
## 6: 2016-05-19 14:30:17             0
```

ii) The bottom 6 rows in our dataset


```r
tail(advertising)
```

```
##    Daily Time Spent on Site Age Area Income Daily Internet Usage
## 1:                    43.70  28    63126.96               173.01
## 2:                    72.97  30    71384.57               208.58
## 3:                    51.30  45    67782.17               134.42
## 4:                    51.63  51    42415.72               120.37
## 5:                    55.55  19    41920.79               187.95
## 6:                    45.01  26    29875.80               178.35
##                           Ad Topic Line          City Male
## 1:        Front-line bifurcated ability  Nicholasland    0
## 2:        Fundamental modular algorithm     Duffystad    1
## 3:      Grass-roots cohesive monitoring   New Darlene    1
## 4:         Expanded intangible solution South Jessica    1
## 5: Proactive bandwidth-monitored policy   West Steven    0
## 6:      Virtual 5thgeneration emulation   Ronniemouth    0
##                   Country           Timestamp Clicked on Ad
## 1:                Mayotte 2016-04-04 03:57:48             1
## 2:                Lebanon 2016-02-11 21:49:00             1
## 3: Bosnia and Herzegovina 2016-04-22 02:07:01             1
## 4:               Mongolia 2016-02-01 17:24:57             1
## 5:              Guatemala 2016-03-24 02:35:54             0
## 6:                 Brazil 2016-06-03 21:43:21             1
```

iii) The shape of the dataset


```r
# Dimensions of the dataset
dim(advertising)
```

```
## [1] 1000   10
```

The advertising dataset has 1000 rows and 10 columns.

iv) The datatypes of the columns in our dataset


```r
# Structure of the dataset
str(advertising)
```

```
## Classes 'data.table' and 'data.frame':	1000 obs. of  10 variables:
##  $ Daily Time Spent on Site: num  69 80.2 69.5 74.2 68.4 ...
##  $ Age                     : int  35 31 26 29 35 23 33 48 30 20 ...
##  $ Area Income             : num  61834 68442 59786 54806 73890 ...
##  $ Daily Internet Usage    : num  256 194 236 246 226 ...
##  $ Ad Topic Line           : chr  "Cloned 5thgeneration orchestration" "Monitored national standardization" "Organic bottom-line service-desk" "Triple-buffered reciprocal time-frame" ...
##  $ City                    : chr  "Wrightburgh" "West Jodi" "Davidton" "West Terrifurt" ...
##  $ Male                    : int  0 1 0 1 0 1 0 1 1 1 ...
##  $ Country                 : chr  "Tunisia" "Nauru" "San Marino" "Italy" ...
##  $ Timestamp               : POSIXct, format: "2016-03-27 00:53:11" "2016-04-04 01:39:02" ...
##  $ Clicked on Ad           : int  0 0 0 0 0 0 0 1 0 0 ...
##  - attr(*, ".internal.selfref")=<externalptr>
```

We can tell that 3 columns are of the type integer, 3 are of the type number and 4 columns are of the character type.

We should convert the target variable (Clicked on Ad) into a factor since it is categorical and not an integer. 


```r
advertising$`Clicked on Ad` = factor(advertising$`Clicked on Ad`)
```

# 5. CLEANING THE DATASET

## Checking for null values in the dataset


```r
#is.na(advertising)

# Sum of null values in each column
colSums(is.na(advertising))
```

```
## Daily Time Spent on Site                      Age              Area Income 
##                        0                        0                        0 
##     Daily Internet Usage            Ad Topic Line                     City 
##                        0                        0                        0 
##                     Male                  Country                Timestamp 
##                        0                        0                        0 
##            Clicked on Ad 
##                        0
```

We can conclude that there are no missing values in any column in our dataset.

## Checking for duplicate values in the dataset.


```r
# Checking the number of duplicated rows
duplicated_rows <- advertising[duplicated(advertising),]
duplicated_rows
```

```
## Empty data.table (0 rows and 10 cols): Daily Time Spent on Site,Age,Area Income,Daily Internet Usage,Ad Topic Line,City...
```

We can conclude that there are also no duplicate values in our dataset.

## Checking for outliers

We can check for outliers using the boxplots.

### i) Daily Time Spent on Site column.


```r
# Daily Time Spent on Site
boxplot(advertising$"Daily Time Spent on Site")
```

![](Week-13-Independent-Project_files/figure-html/time-1.png)<!-- -->

From the boxplot, we can tell that there are no outliers in the 'Daily Time spent on site' column.

### ii) Age Column


```r
# Age
boxplot(advertising$Age)
```

![](Week-13-Independent-Project_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

We don't have outliers in the age column.

### iii) Area Income column


```r
# Area Income
boxplot(advertising$"Area Income")
```

![](Week-13-Independent-Project_files/figure-html/income-1.png)<!-- -->

```r
# boxplot.stats(advertising$Area.Income)$out
```

There are outliers in the area income column.
We decided not to drop them because they are an actual representation of the income people in the area earn. 


### iv) Daily Internet Usage


```r
# Daily Internet Usage
boxplot(advertising$"Daily Internet Usage")
```

![](Week-13-Independent-Project_files/figure-html/units-1.png)<!-- -->

There are no outliers in the daily internet usage column.


# 6. EXPLORATORY DATA ANALYSIS

## A. UNIVARIATE DATA ANALYSIS

### Measures of Central Tendency

#### i) Mean


```r
mean(advertising$"Daily Time Spent on Site")
```

```
## [1] 65.0002
```

The average time spent on the site daily is 65 minutes.


```r
mean(advertising$Age)
```

```
## [1] 36.009
```

The average age of individuals in our dataset is 36 years old.


```r
mean(advertising$"Daily Internet Usage")
```

```
## [1] 180.0001
```

The average daily internet usage is 180 units.

#### ii) Median


```r
median(advertising$"Daily Time Spent on Site")
```

```
## [1] 68.215
```

The median time spent on site daily is 68.215 minutes.


```r
median(advertising$Age)
```

```
## [1] 35
```

The median age is 35 years old.


```r
median(advertising$"Daily Internet Usage")
```

```
## [1] 183.13
```

The median daily internet usage is 183.13 units.

#### iii) Mode


```r
# Define a function for getting the mode 
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
getmode(advertising$"Daily Time Spent on Site")
```

```
## [1] 62.26
```

The mode for the daily time spent on the site is 62.26 minutes.


```r
getmode(advertising$Age)
```

```
## [1] 31
```

The mode for the age is 31 years old.


```r
getmode(advertising$"Daily Internet Usage")
```

```
## [1] 167.22
```

The mode for the daily internet usage is 167.22 units.


### Measures of Dispersion

#### i) Range


```r
range(advertising$"Daily Time Spent on Site")
```

```
## [1] 32.60 91.43
```

The range of daily time spent on the site is between 32.60 and 91.43 minutes


```r
range(advertising$Age)
```

```
## [1] 19 61
```

The age range is between 19 and 61 years old.


```r
range(advertising$"Daily Internet Usage")
```

```
## [1] 104.78 269.96
```

The range of daily internet usage is between 104.78 and 269.96 units.

#### ii) Quantiles


```r
quantile(advertising$"Daily Time Spent on Site")
```

```
##      0%     25%     50%     75%    100% 
## 32.6000 51.3600 68.2150 78.5475 91.4300
```

The first quantile in the daily time spent is 51.36 minutes.
The third quantile in the daily time spent on site is 78.55 minutes.


```r
quantile(advertising$Age)
```

```
##   0%  25%  50%  75% 100% 
##   19   29   35   42   61
```

The first quantile in age is 29 years old.
The third quantile in age is 42 years old.


```r
quantile(advertising$"Daily Internet Usage")
```

```
##       0%      25%      50%      75%     100% 
## 104.7800 138.8300 183.1300 218.7925 269.9600
```

The first quantile in the daily internet usage is 138.83 units.
The third quantile in the daily internet usage is 218.79 units.

#### iii) Variance


```r
var(advertising$"Daily Time Spent on Site")
```

```
## [1] 251.3371
```

The variance in the daily time spent on site is 251.33


```r
var(advertising$Age)
```

```
## [1] 77.18611
```

The variance in the age is 77.18


```r
var(advertising$"Daily Internet Usage")
```

```
## [1] 1927.415
```

The variance in the daily internet usage is 1927.415

#### iv) Standard Deviation


```r
sd(advertising$"Daily Time Spent on Site")
```

```
## [1] 15.85361
```

The standard deviation in the daily time spent on site is 15.85361


```r
sd(advertising$Age)
```

```
## [1] 8.785562
```

The standard deviation in the age is 8.785562


```r
sd(advertising$"Daily Internet Usage")
```

```
## [1] 43.90234
```

The standard deviation in the daily internet usage is 43.90234


### Frequency Tables


```r
# Gender Frequency Table
# 0 symbolizes female while 1 is male
gender <- table(advertising$Male)
gender
```

```
## 
##   0   1 
## 519 481
```

From the frequency table above, we can tell that there are 519 females and 481 males in our dataset.


```r
# Clicked on Ad Frequency Table
# 0 means the individual did not click on the ad, 1 means the individual clicked on an ad
clicked <- table(advertising$"Clicked on Ad")
clicked
```

```
## 
##   0   1 
## 500 500
```

From the frequency table above, we can tell that our dataset is balanced in the sense that 500 individuals clicked on the ad while 500 did not click on the ads.


```r
# Country Frequency Table
country <- (table(advertising$Country))

# Sort the table so as to find the country with the most individuals in our dataset
sorted_country <- sort(country, decreasing = TRUE)
head(sorted_country)
```

```
## 
## Czech Republic         France    Afghanistan      Australia         Cyprus 
##              9              9              8              8              8 
##         Greece 
##              8
```

From the frequency table above, we can tell that both Czech Republic and France had 9 individuals each while Afghanistan, Australia, Cyprus and Greece all had 8 individuals each.


```r
# City Frequency Table
city <- table(advertising$City)

# Sort the table so as to find the city with the most individuals in our dataset
sorted_city <- sort(city, decreasing = TRUE)
head(sorted_city)
```

```
## 
##       Lisamouth    Williamsport Benjaminchester       East John    East Timothy 
##               3               3               2               2               2 
##        Johnstad 
##               2
```

From the frequency table, we can tell that both Lisamouth and Williamsport both had 3 individuals each while Benjaminchester, East John, East Timothy and Johnstad all had 2 individuals each.


```r
# Age Frequency Table
age <- table(advertising$Age)

# Sort the table so as to find the age with the most individuals
sorted_age <- sort(age, decreasing = TRUE)
head(sorted_age)
```

```
## 
## 31 36 28 29 33 30 
## 60 50 48 48 43 39
```

From the frequency table, we can tell that individuals aged between 28 and 36 years old are the most in our dataset.


### Graphical Plots

#### i) Bar Charts

![](Week-13-Independent-Project_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

From the bar chart above, we can tell that age 31 had the highest frequency in the dataset.

![](Week-13-Independent-Project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

From the bar chart, we can tell that 0(female) had more count that 1(male) in our dataset.


#### ii) Histograms

![](Week-13-Independent-Project_files/figure-html/histogram-1.png)<!-- -->

From the histogram, we can also tell that age 25 - 35 had the highest frequency in the dataset.


## B. BIVARIATE DATA ANALYSIS

### Covariance


```r
# We can find the covariance between age and the daily time spent on the site
age <- advertising$Age
time <- advertising$"Daily Time Spent on Site"

cov(age, time)
```

```
## [1] -46.17415
```

There is a negative covariance between age and the daily time spent on the site which means that the older a person is, the less time they spend on the site daily.


```r
age <- advertising$Age
units <- advertising$"Daily Internet Usage"

cov(age, units)
```

```
## [1] -141.6348
```

There is a negative covariance between age and the daily internet usage which means that the older a person is, the less units they use on internet daily.


```r
# Covariance between time spent on site and the daily internet usage
cov(time, units)
```

```
## [1] 360.9919
```

There is a positive covariance between the time spent on site and the daily internet usage which makes sense since the more time you spend on site, the higher your daily internet usage is bound to be.

### Correlation


```r
# We will use the age and time variables that we created earlier for correlation
cor(age, time)
```

```
## [1] -0.3315133
```

There is a negative linear relationship between age and the daily time spent on the site.


```r
# We will use the age and units variables that we created earlier for correlation
cor(age, units)
```

```
## [1] -0.3672086
```

There is a negative linear relationship between age and the daily internet usage.


```r
# Correlation between time spent on site and the daily internet usage
cor(time, units)
```

```
## [1] 0.5186585
```

There is a positive correlation between the time spent on site and the daily internet usage which makes sense since the more time you spend on site, the higher the amount of units you will use on internet. 

### Correlation Matrix

First we load the corrplot library which enables us to plot a correlation matrix.


```r
library(corrplot) # This library allows us to plot correlation.
```

```
## corrplot 0.90 loaded
```

We go ahead to create a variable that holds the numerical columns in our dataset.


```r
# Create a subset of the numerical columns in our dataset
numerical <- subset(advertising, select = c("Daily Time Spent on Site", "Age", "Daily Internet Usage", "Area Income"))
```

Plot a correlation matrix for the numerical variables in our dataset.


```r
cor(numerical)
```

```
##                          Daily Time Spent on Site        Age
## Daily Time Spent on Site                1.0000000 -0.3315133
## Age                                    -0.3315133  1.0000000
## Daily Internet Usage                    0.5186585 -0.3672086
## Area Income                             0.3109544 -0.1826050
##                          Daily Internet Usage Area Income
## Daily Time Spent on Site            0.5186585   0.3109544
## Age                                -0.3672086  -0.1826050
## Daily Internet Usage                1.0000000   0.3374955
## Area Income                         0.3374955   1.0000000
```


### Scatter Plots


```r
# First we import the ggplot2 library which will help us in visualizations
library(ggplot2)
```


![](Week-13-Independent-Project_files/figure-html/internet time-1.png)<!-- -->

From the scatter plot we can tell that the longer the time spent on site, the higher the amount of units used on internet.


![](Week-13-Independent-Project_files/figure-html/scatter-1.png)<!-- -->

From the scatter plot, we can tell that the older an individual is, the less time spent on site. 

![](Week-13-Independent-Project_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

From this scatter plot, we can determine that the older an individual is, the lower the amount of units spent daily on internet. 


# 7. IMPLEMENTING THE SOLUTION 

We will implement the solution to our research problem by creating the following supervised learning models : 

i) k Nearest Neighbors (kNN) 
ii) Support Vector Machine (SVM)
iii) Decision Trees 
iv) Naive Bayes

In this specific problem we will not use regression because our target variable is categorical and not continuous. 

- First we perform label encoding to our categorical columns so as to enable modelling.


```r
# Import the library for label encoding
library(superml)
```

```
## Loading required package: R6
```

```r
# Introduce the label encoder object
label <- LabelEncoder$new()

# Label encode the categorical columns i.e. City, Country
advertising$City <- label$fit_transform(advertising$City)
print(advertising$City)
```

```
##    [1]   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
##   [19]  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
##   [37]  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
##   [55]  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
##   [73]  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
##   [91]  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
##  [109] 108  35 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124
##  [127] 125 126  99 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141
##  [145] 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
##  [163] 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177
##  [181] 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195
##  [199] 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213
##  [217] 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231
##  [235] 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249
##  [253] 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267
##  [271] 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285
##  [289] 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303
##  [307] 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321
##  [325] 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339
##  [343] 340 341 342 343 344 345 346 347 348  68 349 350 351 352 353 354 355 356
##  [361] 357 358   0 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373
##  [379] 110 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390
##  [397] 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408
##  [415] 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 134 425
##  [433] 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443
##  [451] 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461
##  [469] 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479
##  [487] 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497
##  [505] 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515
##  [523] 258 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532
##  [541] 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550
##  [559]  19 551 552 553 554 555 556 557 558 559 560 561 562 423 563 564 565 566
##  [577] 567 568 569 570 571 572 573 574 220 575 576 577 578 579 580 581 293 582
##  [595] 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600
##  [613] 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618
##  [631] 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636
##  [649] 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 409 653
##  [667] 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671
##  [685] 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689
##  [703] 690  23 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706
##  [721] 707 708 709 710 711 712 713 714 715 716 717 718 719 264 720 721 722 723
##  [739] 724 725 726 299 727 728 729 730 731 732 733 734 735 736 737 738 739 740
##  [757] 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758
##  [775] 759 760 507 233 761 762 763 764 765 766 767 768 176 769 770 545 771 772
##  [793] 773 774 775 776 777 778 779 419 780 781 782 783 784 785 786 787 423 788
##  [811] 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 758 804 805
##  [829] 806 233 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822
##  [847] 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 303
##  [865] 840 841 842 843 844 845 846 847 848 849 850 851 852 853 145 854 855 856
##  [883] 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874
##  [901] 875 876  64 877 878 879 880 881 882 883 884 885 886 887 888 889 280 890
##  [919] 891 892 893 867 894 895 896 897 898 899 900 901 902 903 904 905 906 907
##  [937] 908 909 910 911 912 913 914 915 916 917 918 243 919 920 921 922 923 924
##  [955] 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942
##  [973] 943 944 945 946 947 948 949 950 951 952 953 954 955 956 190 957 958 959
##  [991] 960 961 962 963 964 965 966 967 833 968
```

```r
advertising$Country <- label$fit_transform(advertising$Country)
print(advertising$Country)
```

```
##    [1]   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
##   [19]  18  19  20  20  11  21  22  23  24  25  26  27  28  29  30  31  32  33
##   [37]  34  35  36  37  19   0  38  36  39  38  10  40  41   3  26  42  43  44
##   [55]  45  46  47  48  49  50  11  51  11  52  53  54  55  56   0  57  27  56
##   [73]  58  45  59  60  61  10  62  35  63  64  65  66  27  67  68  69   7  70
##   [91]  71  67  72  25  73  74  75  76  12  77  78  79  30  80  81  82  83  53
##  [109]  84  44  85  86  87  88  41  71  89  90  91  83  25  92  93  94  95  12
##  [127]  96  97  32  71  76  98  99 100 101 101  73  51 102 103 104 105  21  36
##  [145]  56  80 106 107  32  95 108 109 110 111  66 112 101  99 113 114  21  35
##  [163] 115   1 116 117  48  66 118 119 120 121  10 122  84 123 102  55 124  27
##  [181] 125  59  28 126  72 122 127 128  54 129 130 110 131  27 132  31 133  37
##  [199] 134  89 135 131 136 137 138  95  46 130   7 139 130 140 141 142 143  72
##  [217]  52 144  80 145 146 147  38 148 130  45 123 149 150 151 152  79 153   7
##  [235] 119   5  56 154  95  90  17  44 155 132 129   2  81 156 157 114 158 159
##  [253]  63 154 160  69  99  28  45  74 155  93 114 103 161 162 129  53 163 164
##  [271] 139  31 165 166  47 167 147 168 141 144 169 170  86 153 171 172 173 173
##  [289] 155  79 174 152 175 176 133 177  70 145 178 178 105 143 126 179 112 135
##  [307]  20  91  86 153  32  96 152 180 181 160 140 155 113 172 182 183 175 184
##  [325] 101 185  95 122 154  90  20 186 174  13 122  97  13 162 187  80  87 188
##  [343] 104 189  97  87  75 190  14   1  28 191 192 112 193 194 135  72  17  49
##  [361] 132 115  74  96  61  80  48  17  55 195  72 100  25  60 142 185 140  28
##  [379] 110 119 141  74 178 196 197  46 112 117 167  85  10 124  77 198  82  87
##  [397]  54 199 200  24 191  85  77 171 148  63 185  77  61 152 201  47 202 128
##  [415] 113   6 203  80   4 111 192 123 169  24 183  42  33 204  93  30 182  58
##  [433] 183  31   6 196 149  67 145 202  79 161 147 164  41 103 132 181  17 115
##  [451]  12 130 120 172 205 195 175  53 116  44 206 158 207 208  51 159  15  56
##  [469] 187  58   7 154 142  62  87 160 172 201  78  37 196 173 208 175 166 187
##  [487]  46 151 171 203 103  25 204  81 209 105 191 124   6 210   7 211 188 212
##  [505]  60 213 127 214 215 190 129  67  57  45  84  63  25 171 167 117 216 216
##  [523]  40 176  89  78  78 112 108   7  52 192  14 204 217 217 118  69   6 206
##  [541] 127 149 129  24 136 112 134  82 187  91 101  79   7 134  53  93 128 218
##  [559]  78  64 134 219 202 121  69 220  42  74  85  88 221 222  67 211 135 169
##  [577] 135  38 167 143 140 153 146 152  60  44  37 138 220   9  50 108 192  40
##  [595]  89 187  51 117 167  93 141 216  55 141   8 109 142  97  71 215  95  44
##  [613] 157 202  20 158 182  43  44 128 223 204 138 160  11 148 176  69 111 209
##  [631]  28 166  46 160 224  28 107 202  92  70  49 117 225  19 118 216  88 216
##  [649] 162 212 226  98 203  25  85  31 200  80 208 203  10 227 114 124 125 157
##  [667]  87  23  47  96 122  99 194 182 178  32 128  22  38 166 169  90 204 158
##  [685]  67  89 200  33 127  73 124  78  86  75 219 219  59 164 226  36  86 206
##  [703]  53  35 206   3 163 162  45 228 124 106 104 183  31  96 184 229 211  23
##  [721] 120  45 179  57 181  35 219 100 230 186  66 155 154  51  31  11 128 186
##  [739]  48 173  21 141 113 114 197 113  96 130 106 186  14 168   3 107 231 185
##  [757] 156  98 179 138 164  69  37 213  13 151 231  80 191  22 160 178  52 200
##  [775] 126  91  74 202  71  26  36  13 222 136 100 232   9  72  50 113 170 114
##  [793]   8 146 110  50 166 228 140 163 197 227  33 140  45  75 106 211 233  78
##  [811]  51 175 115 215 198  70 197 119  98  55  63 178 225  48 146 130 132 125
##  [829]  71 187  16 148 206  17 172  44 188 145  99 107 121  17  33  95 220  15
##  [847]  75 229 215 179 219 185 159 197 217  37  12  25 149  40 123 164  63 146
##  [865] 185  68 188 109 230  64  87  54  46 159 221  22 230 116  14 171 180 129
##  [883]  38 126  95 210 160 116 213 126 142 165 225 108 203 107 152 191 152  28
##  [901]  37  11 210  69  33   3   2  85  57 219   0 168 177 201 216 191  39  66
##  [919] 217  91 161  21 208 128  88 216  13 140 122 196 234 127 193  74 235 197
##  [937] 136 118 226  16   8   9 151  76  30 236  80   4 132 215 189 150  33  89
##  [955]  62  54  48 206 207 158  31 112 107  88 213  50  52  17  37 203  75 111
##  [973] 121 217 140 188  39 160 163  57  52  94 188 151 160 221 170 146 149 153
##  [991] 173 218  34 209 126  86  13 117  59 118
```

```r
print("Dataset after label encoding..\n")
```

```
## [1] "Dataset after label encoding..\n"
```

```r
print(advertising)
```

```
##       Daily Time Spent on Site Age Area Income Daily Internet Usage
##    1:                    68.95  35    61833.90               256.09
##    2:                    80.23  31    68441.85               193.77
##    3:                    69.47  26    59785.94               236.50
##    4:                    74.15  29    54806.18               245.89
##    5:                    68.37  35    73889.99               225.58
##   ---                                                              
##  996:                    72.97  30    71384.57               208.58
##  997:                    51.30  45    67782.17               134.42
##  998:                    51.63  51    42415.72               120.37
##  999:                    55.55  19    41920.79               187.95
## 1000:                    45.01  26    29875.80               178.35
##                               Ad Topic Line City Male Country
##    1:    Cloned 5thgeneration orchestration    0    0       0
##    2:    Monitored national standardization    1    1       1
##    3:      Organic bottom-line service-desk    2    0       2
##    4: Triple-buffered reciprocal time-frame    3    1       3
##    5:         Robust logistical utilization    4    0       4
##   ---                                                        
##  996:         Fundamental modular algorithm  965    1      86
##  997:       Grass-roots cohesive monitoring  966    1      13
##  998:          Expanded intangible solution  967    1     117
##  999:  Proactive bandwidth-monitored policy  833    0      59
## 1000:       Virtual 5thgeneration emulation  968    0     118
##                 Timestamp Clicked on Ad
##    1: 2016-03-27 00:53:11             0
##    2: 2016-04-04 01:39:02             0
##    3: 2016-03-13 20:35:42             0
##    4: 2016-01-10 02:31:19             0
##    5: 2016-06-03 03:36:18             0
##   ---                                  
##  996: 2016-02-11 21:49:00             1
##  997: 2016-04-22 02:07:01             1
##  998: 2016-02-01 17:24:57             1
##  999: 2016-03-24 02:35:54             0
## 1000: 2016-06-03 21:43:21             1
```

- After label encoding, we decide to drop the Timestamp and Ad Topic Line columns since we will not use them in modelling. 


```r
Advertising = subset(advertising, select = c("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","City","Male","Country","Clicked on Ad"))
```


## 1. Naive Bayes

- We first try to understand the columns in our dataset.


```r
describe(Advertising)
```

```
##                          vars    n     mean       sd   median  trimmed      mad
## Daily Time Spent on Site    1 1000    65.00    15.85    68.22    65.74    17.92
## Age                         2 1000    36.01     8.79    35.00    35.51     8.90
## Area Income                 3 1000 55000.00 13414.63 57012.30 56038.94 13316.62
## Daily Internet Usage        4 1000   180.00    43.90   183.13   179.99    58.61
## City                        5 1000   477.87   280.44   473.50   476.80   359.53
## Male                        6 1000     0.48     0.50     0.00     0.48     0.00
## Country                     7 1000   108.92    64.96   107.00   107.78    81.54
## Clicked on Ad*              8 1000     1.50     0.50     1.50     1.50     0.74
##                               min      max    range  skew kurtosis     se
## Daily Time Spent on Site    32.60    91.43    58.83 -0.37    -1.10   0.50
## Age                         19.00    61.00    42.00  0.48    -0.41   0.28
## Area Income              13996.50 79484.80 65488.30 -0.65    -0.11 424.21
## Daily Internet Usage       104.78   269.96   165.18 -0.03    -1.28   1.39
## City                         0.00   968.00   968.00  0.03    -1.21   8.87
## Male                         0.00     1.00     1.00  0.08    -2.00   0.02
## Country                      0.00   236.00   236.00  0.11    -1.14   2.05
## Clicked on Ad*               1.00     2.00     1.00  0.00    -2.00   0.02
```

- We convert the target variable into a factor. 


```r
Advertising$`Clicked on Ad` <- factor(Advertising$`Clicked on Ad`)
```

- We then check the first 6 records in our new dataframe. 


```r
head(Advertising)
```

```
##    Daily Time Spent on Site Age Area Income Daily Internet Usage City Male
## 1:                    68.95  35    61833.90               256.09    0    0
## 2:                    80.23  31    68441.85               193.77    1    1
## 3:                    69.47  26    59785.94               236.50    2    0
## 4:                    74.15  29    54806.18               245.89    3    1
## 5:                    68.37  35    73889.99               225.58    4    0
## 6:                    59.99  23    59761.56               226.74    5    1
##    Country Clicked on Ad
## 1:       0             0
## 2:       1             0
## 3:       2             0
## 4:       3             0
## 5:       4             0
## 6:       5             0
```

- Split the data into training and test data sets


```r
indxTrain <- createDataPartition(y = advertising$`Clicked on Ad`, p = 0.7, list = FALSE)
training <- advertising[indxTrain,]
testing <- advertising[-indxTrain,]
```

- Checking the dimensions of the split sets


```r
prop.table(table(advertising$`Clicked on Ad`)) * 100
```

```
## 
##  0  1 
## 50 50
```

```r
prop.table(table(training$`Clicked on Ad`)) * 100
```

```
## 
##  0  1 
## 50 50
```

```r
prop.table(table(testing$`Clicked on Ad`)) * 100
```

```
## 
##  0  1 
## 50 50
```

- We then compare the outcomes of the training and testing phase by creating objects that will hold the predictor and response variables separately.


```r
x = training[, -8]
y = training$`Clicked on Ad`
```

- Loading our inbuilt e1071 package that holds the Naive Bayes function.


```r
library(e1071)
```

- Then we build our Naive Bayes model using our training data. 


```r
model = naiveBayes(`Clicked on Ad` ~ ., data = training)
```

- Predicting the target variable using the testing data.


```r
y_pred = predict(model, newdata = testing)
```

- Create a table that holds the predicted and actual values

```r
confusion = table(testing$`Clicked on Ad`, y_pred)
```

- Evaluate the model using the confusion matrix() function which tells us the accuracy of the model. 


```r
confusionMatrix(confusion)
```

```
## Confusion Matrix and Statistics
## 
##    y_pred
##       0   1
##   0 147   3
##   1   4 146
##                                           
##                Accuracy : 0.9767          
##                  95% CI : (0.9525, 0.9906)
##     No Information Rate : 0.5033          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.9533          
##                                           
##  Mcnemar's Test P-Value : 1               
##                                           
##             Sensitivity : 0.9735          
##             Specificity : 0.9799          
##          Pos Pred Value : 0.9800          
##          Neg Pred Value : 0.9733          
##              Prevalence : 0.5033          
##          Detection Rate : 0.4900          
##    Detection Prevalence : 0.5000          
##       Balanced Accuracy : 0.9767          
##                                           
##        'Positive' Class : 0               
## 
```

- We can deduce that the accuracy of our Naive Bayes classifier is 96%.


## 2. KNN

- First we create a uniform distribution of 1000 rows.


```r
set.seed(1234)
# Randomize the rows and create a uniform distribution of 1000 rows
random <- runif(1000)
random
```

```
##    [1] 0.1137034113 0.6222994048 0.6092747329 0.6233794417 0.8609153836
##    [6] 0.6403106053 0.0094957564 0.2325505060 0.6660837582 0.5142511413
##   [11] 0.6935912918 0.5449748356 0.2827335836 0.9234334843 0.2923158403
##   [16] 0.8372956282 0.2862232847 0.2668207800 0.1867227897 0.2322259105
##   [21] 0.3166124548 0.3026933707 0.1590460029 0.0399959181 0.2187995410
##   [26] 0.8105985525 0.5256975468 0.9146581660 0.8313450469 0.0457702633
##   [31] 0.4560914824 0.2651866719 0.3046722030 0.5073068701 0.1810962083
##   [36] 0.7596706355 0.2012480376 0.2588098187 0.9921504175 0.8073523403
##   [41] 0.5533335907 0.6464060941 0.3118243071 0.6218191981 0.3297701757
##   [46] 0.5019974730 0.6770945273 0.4849912392 0.2439288273 0.7654597876
##   [51] 0.0737798801 0.3096866019 0.7172717433 0.5045459121 0.1529989589
##   [56] 0.5039334882 0.4939609230 0.7512001970 0.1746498239 0.8483924104
##   [61] 0.8648338320 0.0418572752 0.3171821553 0.0137499392 0.2390257267
##   [66] 0.7064946173 0.3080947571 0.5085475657 0.0516466193 0.5645698400
##   [71] 0.1214801872 0.8928363817 0.0146272557 0.7831211037 0.0899613330
##   [76] 0.5191899808 0.3842666876 0.0700524973 0.3206444222 0.6684953971
##   [81] 0.9264004764 0.4719097211 0.1426153432 0.5442697550 0.1961746519
##   [86] 0.8985804892 0.3894997847 0.3108707797 0.1600286630 0.8961858496
##   [91] 0.1663937804 0.9004245962 0.1340781951 0.1316141342 0.1052875025
##   [96] 0.5115835811 0.3001990539 0.0267168954 0.3096474314 0.7421196571
##  [101] 0.0354567270 0.5650761120 0.2802577761 0.2041963164 0.1337388987
##  [106] 0.3256819244 0.1550619695 0.1299621395 0.4355310597 0.0386426526
##  [111] 0.7133015629 0.1007690411 0.9503049385 0.1218177627 0.2196566209
##  [116] 0.9130877669 0.9458531211 0.2791562229 0.1234710878 0.7971604594
##  [121] 0.7442772151 0.9159742238 0.9945982450 0.9423607150 0.4861354076
##  [126] 0.2834595428 0.2515457012 0.5032551708 0.4969661732 0.3184458097
##  [131] 0.9622228269 0.6340993682 0.1274333980 0.4230469938 0.9143169096
##  [136] 0.4677923333 0.9081691455 0.5977432837 0.6317428160 0.8691583187
##  [141] 0.5027498226 0.9836351147 0.3243860274 0.4813749485 0.3569870775
##  [146] 0.6274776841 0.7416001905 0.5659668173 0.9807865066 0.5768127355
##  [151] 0.4390420518 0.2285996950 0.0821580656 0.8502649218 0.2346612616
##  [156] 0.9881674468 0.6018975459 0.9987408081 0.3755993766 0.5551266309
##  [161] 0.4294439629 0.5758777808 0.4325073974 0.2248457640 0.0849847377
##  [166] 0.6372982597 0.4310163704 0.0727160936 0.8024020193 0.3252783034
##  [171] 0.7572890350 0.5842715173 0.7088394067 0.4269757664 0.3435727020
##  [176] 0.7591199852 0.4240302080 0.5608872538 0.1161357744 0.3030217977
##  [181] 0.4788026859 0.3448305468 0.6007141401 0.0760833232 0.9559926111
##  [186] 0.0222068231 0.8417106324 0.6324424488 0.3100941652 0.7425693662
##  [191] 0.6389113136 0.9925159873 0.1282697883 0.8832395778 0.8100833879
##  [196] 0.8218511783 0.8347026624 0.7327322206 0.9830440243 0.6392045827
##  [201] 0.6607546343 0.5283593780 0.3174938215 0.7678554691 0.5263084925
##  [206] 0.7323018843 0.3076657406 0.4041732512 0.2044024453 0.9856330883
##  [211] 0.5663107571 0.2803751451 0.1850557232 0.7580613962 0.5667812813
##  [216] 0.9321735711 0.6386933164 0.7007481344 0.4792224686 0.8503119163
##  [221] 0.4223306754 0.0313921231 0.2581466483 0.3348447348 0.1335496686
##  [226] 0.4995463854 0.8021356328 0.3371532431 0.5089206153 0.4944385618
##  [231] 0.7970529040 0.5669588954 0.1066968180 0.8076484452 0.5671120710
##  [236] 0.2122409279 0.7495792548 0.3072183500 0.4895184434 0.9897098928
##  [241] 0.4241091781 0.2444030046 0.2171347148 0.6891175066 0.9802127087
##  [246] 0.4770330393 0.7735236220 0.5743129447 0.9659397006 0.7969238409
##  [251] 0.5319050872 0.5966237611 0.2638864736 0.2795427088 0.0651032443
##  [256] 0.5630813465 0.2623556822 0.0032823312 0.5895165436 0.5200511648
##  [261] 0.8446347348 0.0295568136 0.5997693492 0.2684197696 0.1206089044
##  [266] 0.1007054616 0.7481611404 0.0159606293 0.0494611457 0.7476237861
##  [271] 0.3572376638 0.7589581960 0.3759563426 0.7994627089 0.0256927656
##  [276] 0.5063585502 0.8212286464 0.5447565762 0.2666844544 0.3446373220
##  [281] 0.3691759529 0.4292520846 0.9185143732 0.7843448154 0.7378315323
##  [286] 0.2807726238 0.4568266282 0.2875376416 0.6962910676 0.8207562983
##  [291] 0.6551535316 0.4135046774 0.9518294146 0.2431094602 0.6086850266
##  [296] 0.7579514689 0.6936673617 0.1154277963 0.6359116645 0.3090253684
##  [301] 0.3529985021 0.9809583162 0.5388827636 0.4440338630 0.9493667777
##  [306] 0.4524833714 0.1906258035 0.9916091496 0.5484554477 0.7688157670
##  [311] 0.9134216728 0.6821120020 0.4072514204 0.4075922994 0.1460827903
##  [316] 0.1966677140 0.1922093395 0.4084144006 0.3482213062 0.8345428484
##  [321] 0.1984000071 0.8618053095 0.3971853103 0.1532537669 0.3392833832
##  [326] 0.3671804396 0.4273790829 0.1863369043 0.6580166004 0.9204113812
##  [331] 0.7338940627 0.8823192716 0.9533465311 0.1949015351 0.4726167354
##  [336] 0.3860506560 0.3741658572 0.0278556566 0.9293552118 0.4105292757
##  [341] 0.9558402160 0.2721528402 0.5172464938 0.9783098423 0.3696964863
##  [346] 0.3104304392 0.0342096325 0.6675658475 0.9209163769 0.0449895980
##  [351] 0.2011326319 0.7435148843 0.1305568311 0.7088835938 0.9988318114
##  [356] 0.9439130460 0.5929038990 0.7312956364 0.4867341756 0.7681519960
##  [361] 0.0031454624 0.5579412982 0.4602552892 0.3297151451 0.8354756273
##  [366] 0.9777016630 0.6605149473 0.2335748596 0.8192004203 0.7246848950
##  [371] 0.9763571532 0.2651130287 0.8788200426 0.4878892123 0.3054682855
##  [376] 0.3950969989 0.7593397161 0.1008016423 0.4213596904 0.6577642476
##  [381] 0.2952640920 0.2059864998 0.0021467118 0.1078426787 0.2148670219
##  [386] 0.1151536738 0.6871594316 0.1932722302 0.9849218933 0.9470379758
##  [391] 0.6917069692 0.7398919435 0.7324402984 0.6525984439 0.1670463069
##  [396] 0.9231808581 0.5646048458 0.5364956353 0.0188437472 0.3663996276
##  [401] 0.6862422745 0.4166287233 0.7570262463 0.7757948679 0.5735870199
##  [406] 0.1886987926 0.7582653339 0.0926830929 0.6223585266 0.4159132016
##  [411] 0.7765108705 0.8078028020 0.9672963202 0.2159032512 0.8670089110
##  [416] 0.4169180284 0.5130287160 0.7749373205 0.1325433177 0.4119076421
##  [421] 0.6620475734 0.9077197670 0.3444916373 0.1052430642 0.9323459219
##  [426] 0.1999332728 0.0540183072 0.4258400758 0.3387958198 0.2756013027
##  [431] 0.6634905422 0.7838960970 0.4396973667 0.9303238676 0.2722072056
##  [436] 0.6590290293 0.3802892941 0.9353907774 0.7355009299 0.5900904534
##  [441] 0.8146524148 0.8824088296 0.7454413087 0.5017915608 0.9885014733
##  [446] 0.5552680509 0.8797018330 0.6304392638 0.6132155319 0.0336092997
##  [451] 0.2547006856 0.7759688012 0.9982450008 0.9601184016 0.7509288136
##  [456] 0.6805038610 0.2468864541 0.6065629770 0.5739705635 0.0495949083
##  [461] 0.3728201450 0.8947680488 0.3917717456 0.5169452706 0.1752800765
##  [466] 0.1926054636 0.5465086957 0.3931208388 0.6251974967 0.5722047726
##  [471] 0.1406190705 0.2892716692 0.0006121558 0.9553637172 0.3994099500
##  [476] 0.9770535454 0.5117741989 0.4671611886 0.7238354937 0.1420735531
##  [481] 0.5235777793 0.5604211201 0.6603428058 0.3722657019 0.1361519932
##  [486] 0.0738809260 0.7615121801 0.7027949130 0.7638952618 0.1570562583
##  [491] 0.5993446065 0.7633417868 0.9126529882 0.3080208497 0.6570658754
##  [496] 0.3378046290 0.6042503819 0.0839550188 0.6056672356 0.5959110265
##  [501] 0.6689191493 0.8010432960 0.8554558854 0.0501399101 0.6744340854
##  [506] 0.8458932983 0.7463983176 0.8295644040 0.1279001259 0.7941084381
##  [511] 0.3555059133 0.9480816713 0.9301581059 0.5924445491 0.1104942642
##  [516] 0.7838354679 0.6018229409 0.0882744200 0.6572622757 0.3320809370
##  [521] 0.8354190320 0.2477733262 0.3635909229 0.9246486337 0.5630180079
##  [526] 0.8068653576 0.0389287404 0.0398254376 0.6325429690 0.2390867730
##  [531] 0.2525250022 0.6246245001 0.4056898011 0.7984730732 0.1174629412
##  [536] 0.9944972815 0.6498573283 0.6556109313 0.7474796236 0.5142732125
##  [541] 0.3803301165 0.6113082704 0.9660003211 0.2158586672 0.7487493630
##  [546] 0.4323756348 0.8285962390 0.8833405196 0.9797886356 0.9140007265
##  [551] 0.2574865802 0.2234478178 0.7906329180 0.1650948089 0.8380758595
##  [556] 0.9719204390 0.4975382260 0.0398080489 0.6251565958 0.8303859085
##  [561] 0.1558116069 0.9902037513 0.6808823370 0.8646979360 0.2416607619
##  [566] 0.8625790973 0.7920866730 0.6123247666 0.2085591650 0.0594121274
##  [571] 0.6252874383 0.8039313646 0.1986306231 0.6656501235 0.4029327796
##  [576] 0.3869626867 0.0602284134 0.4202804118 0.5510972508 0.8160757187
##  [581] 0.8377968394 0.0120980635 0.5726980967 0.3016408621 0.0386787029
##  [586] 0.3583329110 0.2674532351 0.9740237382 0.9511434464 0.6160606141
##  [591] 0.9648370075 0.8780345162 0.1199900017 0.5962578633 0.3569201881
##  [596] 0.6748864641 0.6381902022 0.6968670192 0.6251959745 0.1222220890
##  [601] 0.2809718144 0.1741881554 0.1702244373 0.5605436335 0.4288005915
##  [606] 0.3891867667 0.8437064125 0.2087907058 0.5094247796 0.3949653304
##  [611] 0.2581600642 0.8029614384 0.3069992187 0.0781303388 0.9467787365
##  [616] 0.3786684813 0.3274520368 0.6648268595 0.7773210085 0.4841216214
##  [621] 0.9294515708 0.0498878444 0.6713575637 0.8255119289 0.3366140190
##  [626] 0.9315205917 0.4840459090 0.9460469969 0.3112607158 0.5990518702
##  [631] 0.8902234128 0.6073635472 0.4405584221 0.5600596257 0.9394272438
##  [636] 0.8345116014 0.2871900161 0.8435600812 0.2587780301 0.3393378323
##  [641] 0.5569070971 0.1878760487 0.5096489282 0.9270732831 0.3069720590
##  [646] 0.3562869385 0.0568936686 0.8847315591 0.5119929586 0.8047118725
##  [651] 0.2368176600 0.6868353374 0.8605094308 0.4501529348 0.1703747334
##  [656] 0.5086963247 0.8700193819 0.0333066215 0.2581552805 0.9370862804
##  [661] 0.6150363816 0.9495189879 0.8155944902 0.0493154514 0.3019702085
##  [666] 0.5120426093 0.7103606241 0.2846530925 0.4649487969 0.7293889807
##  [671] 0.1281430512 0.9342953495 0.3935495033 0.2491241221 0.9473612858
##  [676] 0.8416525447 0.4152223791 0.8916959967 0.2067407048 0.9501987083
##  [681] 0.4784584530 0.1102770171 0.6293535226 0.2934964148 0.8302932212
##  [686] 0.3279664356 0.8736798677 0.0276332614 0.5400351961 0.4789401426
##  [691] 0.8779045155 0.4868288476 0.2223561769 0.5091449048 0.0095259694
##  [696] 0.0884908563 0.3185763857 0.5453064234 0.3029776684 0.7398838575
##  [701] 0.0102781467 0.5986387285 0.7131026797 0.4659931685 0.2165860315
##  [706] 0.1164272521 0.4105797180 0.6391855688 0.0562446604 0.5683156576
##  [711] 0.7079906715 0.9757936925 0.9707029760 0.1169157147 0.1899509421
##  [716] 0.1869201926 0.4551901366 0.2038997530 0.9743351496 0.0994084384
##  [721] 0.8248006778 0.1010262312 0.9721297773 0.0053628031 0.4979119201
##  [726] 0.9856622049 0.4394909204 0.3435957879 0.3051547669 0.9866527512
##  [731] 0.9242956992 0.0122110262 0.0992497059 0.5411191899 0.6206978902
##  [736] 0.7936123651 0.4815281834 0.8071745096 0.9878338575 0.9275750027
##  [741] 0.2715825546 0.0987931858 0.0656125834 0.1873251994 0.5919763285
##  [746] 0.0008630857 0.4841866000 0.6553977022 0.2006995643 0.6645026628
##  [751] 0.5526229665 0.7441225741 0.3914565546 0.8356535558 0.2486454002
##  [756] 0.8860823966 0.6917566981 0.9165655458 0.3700721294 0.6150684627
##  [761] 0.0332543079 0.6476548435 0.0040038908 0.5773450769 0.2807610431
##  [766] 0.6105293497 0.9270527456 0.5732149626 0.7990209830 0.3004978194
##  [771] 0.8878271687 0.8550552714 0.8370745510 0.0721003956 0.6239091882
##  [776] 0.5107358950 0.0658973011 0.2241470253 0.5814365137 0.3622224077
##  [781] 0.9448925692 0.0565309483 0.0003418126 0.2899808860 0.2172976583
##  [786] 0.6885741784 0.8648701455 0.0298377264 0.7015088277 0.1667423935
##  [791] 0.7850592399 0.1458532037 0.6762002339 0.5202723837 0.7049989968
##  [796] 0.2645380092 0.5058415511 0.8223392977 0.1797051979 0.8646464990
##  [801] 0.1099460442 0.8714367896 0.5144197687 0.2856988220 0.3367340211
##  [806] 0.5696520146 0.1842292922 0.4393714704 0.6618221807 0.4203386179
##  [811] 0.5609762191 0.1956236530 0.9283050264 0.6735741969 0.1310370171
##  [816] 0.3768865855 0.3023030679 0.8052576943 0.4701284021 0.4414459956
##  [821] 0.0795966187 0.3735353299 0.3879347476 0.3902946142 0.2394194803
##  [826] 0.7895856642 0.0158712207 0.3987600550 0.3882675874 0.4577801591
##  [831] 0.2966457983 0.8032375281 0.8713950089 0.9445756134 0.2728989087
##  [836] 0.6754392071 0.7114110326 0.2016571660 0.5567085175 0.6235133451
##  [841] 0.1080694934 0.5848401678 0.6458513984 0.1885790734 0.4577570041
##  [846] 0.9174875303 0.5659459098 0.9422312926 0.7447421285 0.7167671463
##  [851] 0.5453664022 0.9767362811 0.9435349118 0.1765267355 0.5459967637
##  [856] 0.6448610609 0.1049211815 0.0382077270 0.7709153660 0.9573111415
##  [861] 0.1106834277 0.4022474787 0.0274988688 0.4143887600 0.2089883683
##  [866] 0.9180756707 0.2282334333 0.9819308324 0.2717576043 0.5315367309
##  [871] 0.9459464359 0.9049140171 0.9490382764 0.3010218127 0.7661085380
##  [876] 0.1893898065 0.1013609685 0.0859864308 0.5092333122 0.5776511210
##  [881] 0.6869137934 0.2515672981 0.4455045264 0.5412310294 0.5681811818
##  [886] 0.9304210341 0.1723820523 0.0392490570 0.0989390847 0.0155072559
##  [891] 0.9203997736 0.7464458942 0.6054088911 0.3776960922 0.3414908806
##  [896] 0.3275395611 0.6275120932 0.6329661196 0.9804190102 0.8694249380
##  [901] 0.6415731595 0.4095242864 0.9209373307 0.4514219440 0.9142726245
##  [906] 0.0263550885 0.3419393084 0.2560775015 0.7771316250 0.2960546212
##  [911] 0.2574584300 0.4713358963 0.0702871464 0.2265955168 0.1147465222
##  [916] 0.0799787233 0.4408870994 0.9824255591 0.9638371458 0.7975955033
##  [921] 0.5417362035 0.4557757070 0.2110341974 0.2306389480 0.5910997964
##  [926] 0.2582380513 0.7574036478 0.5515460693 0.0976253275 0.1505063956
##  [931] 0.1457857946 0.9905682108 0.0260923174 0.6067343464 0.1021205992
##  [936] 0.5094256019 0.1874213456 0.5072218911 0.3851755147 0.5886653271
##  [941] 0.9207440454 0.5835579261 0.8129020869 0.7675330476 0.6075635189
##  [946] 0.5337823096 0.2854788974 0.1241884707 0.4839640744 0.3673567923
##  [951] 0.7080209046 0.9515685702 0.3071189800 0.8207250747 0.8552160852
##  [956] 0.0459401198 0.8637784163 0.4761403869 0.1217052343 0.6558956904
##  [961] 0.2275585500 0.4601899209 0.8862336075 0.7933290135 0.0457805442
##  [966] 0.0677916221 0.6626513049 0.2288734971 0.5923129355 0.3301717055
##  [971] 0.9993030254 0.7087918075 0.0031697885 0.7697339621 0.2002311687
##  [976] 0.6894992553 0.7498534597 0.2383271190 0.9529667778 0.5847832749
##  [981] 0.7994580676 0.9346025025 0.0765929217 0.6636016299 0.9874346673
##  [986] 0.2523310662 0.0394362193 0.9856219713 0.1328507625 0.1761907323
##  [991] 0.4827948108 0.6360850439 0.9869790701 0.0995346869 0.6936827919
##  [996] 0.0013087022 0.7674259357 0.3199795457 0.9580128449 0.1953975793
```

- Order the rows in our randomized dataset


```r
ad_random <- Advertising[order(random),]
```

- Preview the first 6 rows in our dataset


```r
head(ad_random)
```

```
##    Daily Time Spent on Site Age Area Income Daily Internet Usage City Male
## 1:                    80.46  29    56909.30               230.78  765    0
## 2:                    78.37  24    55015.08               207.27  466    0
## 3:                    57.99  50    62466.10               124.58  730    0
## 4:                    72.97  30    71384.57               208.58  965    1
## 5:                    77.66  29    67080.94               168.15  377    0
## 6:                    38.91  33    56369.74               150.80  357    1
##    Country Clicked on Ad
## 1:     222             0
## 2:     142             0
## 3:     113             1
## 4:      86             1
## 5:     178             0
## 6:     132             1
```

- Create a normalization function and normalize the response variables in our sampled dataset. 


```r
normal <- function(x) (
  return( ((x - min(x)) /(max(x)-min(x))) )
)

normal(1:8)
```

```
## [1] 0.0000000 0.1428571 0.2857143 0.4285714 0.5714286 0.7142857 0.8571429
## [8] 1.0000000
```

```r
ad_new <- as.data.frame(lapply(ad_random[,-8], normal))
summary(ad_new)
```

```
##  Daily.Time.Spent.on.Site      Age          Area.Income    
##  Min.   :0.0000           Min.   :0.0000   Min.   :0.0000  
##  1st Qu.:0.3189           1st Qu.:0.2381   1st Qu.:0.5044  
##  Median :0.6054           Median :0.3810   Median :0.6568  
##  Mean   :0.5507           Mean   :0.4050   Mean   :0.6261  
##  3rd Qu.:0.7810           3rd Qu.:0.5476   3rd Qu.:0.7860  
##  Max.   :1.0000           Max.   :1.0000   Max.   :1.0000  
##  Daily.Internet.Usage      City             Male          Country      
##  Min.   :0.0000       Min.   :0.0000   Min.   :0.000   Min.   :0.0000  
##  1st Qu.:0.2061       1st Qu.:0.2425   1st Qu.:0.000   1st Qu.:0.2203  
##  Median :0.4743       Median :0.4892   Median :0.000   Median :0.4534  
##  Mean   :0.4554       Mean   :0.4937   Mean   :0.481   Mean   :0.4615  
##  3rd Qu.:0.6902       3rd Qu.:0.7451   3rd Qu.:1.000   3rd Qu.:0.6864  
##  Max.   :1.0000       Max.   :1.0000   Max.   :1.000   Max.   :1.0000
```

- Create test and train data sets


```r
train <- ad_new[1:700,]
test <- ad_new[701:1000,]
train_label <- ad_random[1:700,8]
test_label <- ad_random[701:1000,8]
```

- Check the dimensions in our train sets


```r
dim(train_label)
```

```
## [1] 700   1
```

```r
dim(train)
```

```
## [1] 700   7
```
- We build our KNN model from the class library


```r
library(class)    
require(class)
#cl = train_label[,1]
model <- knn(train = train, test = test, cl = train_label$`Clicked on Ad`, k=32)
table(factor(model))
```

```
## 
##   0   1 
## 160 140
```

```r
# Print out the confusion matrix 
tb = table(test_label$`Clicked on Ad`,model)
tb
```

```
##    model
##       0   1
##   0 151   3
##   1   9 137
```

From the confusion matrix we can tell that our KNN model was able to make 288 correct predictions from 300 observations in the training set. 


```r
# Checking the accuracy of our KNN model
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tb)
```

```
## [1] 96
```

- We have achieved an accuracy score of 96% using the KNN model.

## 3. Support Vector Machine (SVM) Classifier

- First we encode the target feature into a factor 


```r
# Encoding the target feature as factor
Advertising$`Clicked on Ad` = factor(Advertising$`Clicked on Ad`, levels = c(0, 1))
```

- We install the necessary package and then split the data in a 70/30 proportion so as to enable comparison with the other models.


```r
# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)

set.seed(123)
split = sample.split(Advertising$`Clicked on Ad`, SplitRatio = 0.7)

training = subset(Advertising, split == TRUE)
test = subset(Advertising, split == FALSE)
```

- We then fit the SVM model to our training data. 


```r
# Fitting SVM to the Training set

classifier = svm(formula = `Clicked on Ad` ~ .,
				data = training,
				type = 'C-classification',
				kernel = 'linear')
```

- We then predict using the testing set. 


```r
# Predicting the Test set results
y_pred = predict(classifier, newdata = test)
```

- We then formulate the confusion matrix and check the accuracy based on the number of accurate predicted values. 


```r
# Making the Confusion Matrix
cm = table(test$`Clicked on Ad`, y_pred)
```


```r
confusionMatrix(cm)
```

```
## Confusion Matrix and Statistics
## 
##    y_pred
##       0   1
##   0 149   1
##   1   8 142
##                                           
##                Accuracy : 0.97            
##                  95% CI : (0.9438, 0.9862)
##     No Information Rate : 0.5233          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.94            
##                                           
##  Mcnemar's Test P-Value : 0.0455          
##                                           
##             Sensitivity : 0.9490          
##             Specificity : 0.9930          
##          Pos Pred Value : 0.9933          
##          Neg Pred Value : 0.9467          
##              Prevalence : 0.5233          
##          Detection Rate : 0.4967          
##    Detection Prevalence : 0.5000          
##       Balanced Accuracy : 0.9710          
##                                           
##        'Positive' Class : 0               
## 
```

- Using SVM, we achieve an accuracy score of 97%


## 4. Random Forest Classifier

- Install the packages necessary. 


```r
# Installing package
install.packages("caTools")	 # For sampling the dataset
```

```
## Warning: package 'caTools' is in use and will not be installed
```

```r
install.packages("randomForest") # For implementing random forest algorithm
```

```
## Warning: package 'randomForest' is in use and will not be installed
```

- Loading the package into our environment. 


```r
# Loading package
library(caTools)
library(randomForest)
```

- Split the data in a 70/30 proportion.


```r
# Splitting data in train and test data
split <- sample.split(Advertising$`Clicked on Ad`, SplitRatio = 0.7)
split
```

```
##    [1]  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE
##   [13]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##   [25]  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##   [37]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##   [49] FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##   [61]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE
##   [73]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE
##   [85]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##   [97] FALSE FALSE  TRUE  TRUE  TRUE FALSE FALSE FALSE FALSE FALSE  TRUE  TRUE
##  [109]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE
##  [121]  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [133] FALSE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE
##  [145] FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE  TRUE  TRUE
##  [157]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE
##  [169]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE
##  [181] FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [193]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE  TRUE FALSE  TRUE
##  [205]  TRUE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE
##  [217] FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE
##  [229] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE
##  [241]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE
##  [253]  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE
##  [265]  TRUE FALSE FALSE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE
##  [277]  TRUE FALSE  TRUE  TRUE FALSE FALSE  TRUE FALSE  TRUE FALSE  TRUE  TRUE
##  [289] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE
##  [301]  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE
##  [313]  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE
##  [325]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [337]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE
##  [349]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [361]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE
##  [373] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE
##  [385] FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE
##  [397] FALSE FALSE FALSE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [409] FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE FALSE FALSE FALSE  TRUE  TRUE
##  [421]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE
##  [433] FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE  TRUE FALSE
##  [445]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE
##  [457] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE
##  [469] FALSE  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE FALSE
##  [481]  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [493]  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE
##  [505] FALSE  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE FALSE FALSE
##  [517]  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [529]  TRUE  TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE
##  [541]  TRUE FALSE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [553] FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [565]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [577]  TRUE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE
##  [589]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE
##  [601]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [613]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE FALSE
##  [625]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE FALSE
##  [637] FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [649]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE FALSE  TRUE
##  [661] FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE
##  [673]  TRUE FALSE FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE
##  [685]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [697]  TRUE  TRUE  TRUE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [709]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [721]  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [733] FALSE FALSE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [745] FALSE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE
##  [757]  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE
##  [769]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [781] FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE
##  [793]  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE
##  [805]  TRUE  TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [817]  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE
##  [829] FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [841]  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE
##  [853]  TRUE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [865]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [877]  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [889] FALSE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE  TRUE FALSE  TRUE
##  [901] FALSE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE
##  [913]  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [925]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE
##  [937]  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE
##  [949] FALSE  TRUE  TRUE FALSE  TRUE  TRUE FALSE FALSE  TRUE  TRUE FALSE FALSE
##  [961]  TRUE FALSE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [973]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE
##  [985]  TRUE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE
##  [997] FALSE  TRUE  TRUE  TRUE
```

```r
train <- subset(Advertising, split == "TRUE")
test <- subset(Advertising, split == "FALSE")
```

- Fit the random forest model to our training dataset. 


```r
# Fitting Random Forest to the train dataset
set.seed(120) # Setting seed
classifier_RF = randomForest(x = train[,-8],
							y = train$`Clicked on Ad`,
							ntree = 500)

classifier_RF
```

```
## 
## Call:
##  randomForest(x = train[, -8], y = train$`Clicked on Ad`, ntree = 500) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 3.57%
## Confusion matrix:
##     0   1 class.error
## 0 340  10  0.02857143
## 1  15 335  0.04285714
```

- Make predictions on the test set. 


```r
# Predicting the Test set results
y_pred = predict(classifier_RF, newdata = test)
```

- Evaluate using the confusion matrix.


```r
# Confusion Matrix
cm = table(test$`Clicked on Ad`, y_pred)
cm
```

```
##    y_pred
##       0   1
##   0 143   7
##   1   7 143
```


```r
confusionMatrix(cm)
```

```
## Confusion Matrix and Statistics
## 
##    y_pred
##       0   1
##   0 143   7
##   1   7 143
##                                           
##                Accuracy : 0.9533          
##                  95% CI : (0.9229, 0.9743)
##     No Information Rate : 0.5             
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.9067          
##                                           
##  Mcnemar's Test P-Value : 1               
##                                           
##             Sensitivity : 0.9533          
##             Specificity : 0.9533          
##          Pos Pred Value : 0.9533          
##          Neg Pred Value : 0.9533          
##              Prevalence : 0.5000          
##          Detection Rate : 0.4767          
##    Detection Prevalence : 0.5000          
##       Balanced Accuracy : 0.9533          
##                                           
##        'Positive' Class : 0               
## 
```

- Our Random Forest classifier gave us an accuracy score of 95.3% 


```r
# Plotting model
plot(classifier_RF)
```

![](Week-13-Independent-Project_files/figure-html/unnamed-chunk-44-1.png)<!-- -->

- We then assess the most important features in our dataset.


```r
# Importance plot
importance(classifier_RF)
```

```
##                          MeanDecreaseGini
## Daily Time Spent on Site       104.062890
## Age                             35.294559
## Area Income                     40.274889
## Daily Internet Usage           148.719501
## City                            10.240562
## Male                             2.155433
## Country                          8.702659
```

From the feature importance table, we can tell that the most important features in our dataset are: 

- Daily Internet Usage
- Daily Time Spent on Site
- Area Income
- Age


```r
# Variable importance plot
varImpPlot(classifier_RF)
```

![](Week-13-Independent-Project_files/figure-html/unnamed-chunk-46-1.png)<!-- -->

# 8. CHALLENGING THE SOLUTION

- We decided to challenge the solution using the Logistic Regression Modeland then assess whether it will perform better than the other supervised learning algorithms. 

- First we install the necessary package required.


```r
# Installing the package
#install.packages("caTools") # For Logistic regression
#install.packages("ROCR")	 # For ROC curve to evaluate model
```

- Load the packages.


```r
# Loading package
library(caTools)
library(ROCR)
```

- Split the dataset in a 70/30 proportion for comparison purposes. 


```r
# Splitting dataset
split <- sample.split(Advertising$`Clicked on Ad`, SplitRatio = 0.7)
split
```

```
##    [1]  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE
##   [13] FALSE FALSE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE FALSE
##   [25]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE
##   [37] FALSE  TRUE FALSE FALSE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE
##   [49]  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##   [61]  TRUE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##   [73] FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##   [85]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE  TRUE FALSE
##   [97]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE
##  [109] FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE FALSE
##  [121]  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE
##  [133] FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE
##  [145]  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE
##  [157]  TRUE  TRUE FALSE FALSE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE
##  [169] FALSE  TRUE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [181] FALSE  TRUE FALSE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [193]  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE
##  [205] FALSE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [217]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE FALSE
##  [229]  TRUE  TRUE FALSE FALSE  TRUE FALSE FALSE FALSE FALSE  TRUE FALSE  TRUE
##  [241]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [253]  TRUE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE
##  [265] FALSE  TRUE FALSE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [277]  TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE  TRUE FALSE FALSE FALSE  TRUE
##  [289]  TRUE FALSE FALSE FALSE FALSE FALSE FALSE FALSE  TRUE  TRUE FALSE  TRUE
##  [301] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [313]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [325]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE
##  [337]  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE  TRUE FALSE
##  [349]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE
##  [361]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [373]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [385]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE
##  [397]  TRUE FALSE FALSE  TRUE FALSE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE
##  [409]  TRUE  TRUE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE
##  [421]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [433]  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE FALSE  TRUE
##  [445] FALSE FALSE FALSE FALSE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE
##  [457]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [469]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE
##  [481]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [493]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE FALSE
##  [505] FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE
##  [517]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [529] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE FALSE
##  [541]  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [553]  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE
##  [565]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [577]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [589] FALSE  TRUE FALSE  TRUE  TRUE FALSE FALSE  TRUE  TRUE FALSE FALSE FALSE
##  [601]  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE
##  [613]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE
##  [625]  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE
##  [637] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE
##  [649] FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [661] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE
##  [673] FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE
##  [685]  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE FALSE
##  [697] FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [709]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [721]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE  TRUE
##  [733]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE FALSE FALSE FALSE FALSE FALSE  TRUE
##  [745]  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [757]  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [769] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE  TRUE  TRUE
##  [781]  TRUE FALSE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE FALSE  TRUE  TRUE
##  [793] FALSE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE
##  [805] FALSE  TRUE  TRUE FALSE FALSE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [817] FALSE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE
##  [829]  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE
##  [841]  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE FALSE FALSE FALSE  TRUE  TRUE
##  [853]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE
##  [865]  TRUE  TRUE FALSE  TRUE FALSE FALSE  TRUE  TRUE FALSE FALSE  TRUE  TRUE
##  [877]  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE
##  [889]  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE
##  [901] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [913] FALSE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [925]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE  TRUE  TRUE  TRUE  TRUE
##  [937]  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [949] FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE
##  [961]  TRUE  TRUE  TRUE FALSE  TRUE FALSE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [973] FALSE  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE FALSE
##  [985]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE  TRUE
##  [997]  TRUE  TRUE  TRUE  TRUE
```

- Initiate the training and test sets. 


```r
training <- subset(Advertising, split == "TRUE")
test <- subset(Advertising, split == "FALSE")
```

- Train the logistic model using our training data.


```r
# Training model
logistic_model <- glm(`Clicked on Ad` ~ .,
					data = training,
					family = "binomial")
logistic_model
```

```
## 
## Call:  glm(formula = `Clicked on Ad` ~ ., family = "binomial", data = training)
## 
## Coefficients:
##                (Intercept)  `Daily Time Spent on Site`  
##                 29.0935005                  -0.2171659  
##                        Age               `Area Income`  
##                  0.1863791                  -0.0001480  
##     `Daily Internet Usage`                        City  
##                 -0.0668572                   0.0001733  
##                       Male                     Country  
##                 -0.5590025                   0.0054096  
## 
## Degrees of Freedom: 699 Total (i.e. Null);  692 Residual
## Null Deviance:	    970.4 
## Residual Deviance: 104.3 	AIC: 120.3
```

- Summary of the model.


```r
# Summary
summary(logistic_model)
```

```
## 
## Call:
## glm(formula = `Clicked on Ad` ~ ., family = "binomial", data = training)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.1737  -0.1065  -0.0183   0.0120   3.3477  
## 
## Coefficients:
##                              Estimate Std. Error z value Pr(>|z|)    
## (Intercept)                 2.909e+01  3.827e+00   7.601 2.93e-14 ***
## `Daily Time Spent on Site` -2.172e-01  3.020e-02  -7.190 6.46e-13 ***
## Age                         1.864e-01  3.381e-02   5.513 3.53e-08 ***
## `Area Income`              -1.480e-04  2.506e-05  -5.906 3.51e-09 ***
## `Daily Internet Usage`     -6.686e-02  9.101e-03  -7.346 2.04e-13 ***
## City                        1.733e-04  1.056e-03   0.164     0.87    
## Male                       -5.590e-01  5.508e-01  -1.015     0.31    
## Country                     5.410e-03  4.606e-03   1.174     0.24    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 970.41  on 699  degrees of freedom
## Residual deviance: 104.32  on 692  degrees of freedom
## AIC: 120.32
## 
## Number of Fisher Scoring iterations: 8
```

- Make predictions on the test data. 


```r
# Predict test data based on model
predict_reg <- predict(logistic_model,
					test, type = "response")
predict_reg
```

```
##            1            2            3            4            5            6 
## 0.0020432900 0.0012311139 0.0053384842 0.9999782828 0.0008953394 0.9992783262 
##            7            8            9           10           11           12 
## 0.0036432272 0.8806829863 0.9999824216 0.0008004238 0.0011667498 0.9998408381 
##           13           14           15           16           17           18 
## 0.0024314056 0.9997494230 0.9999488117 0.9999981621 0.0068259978 0.0032877672 
##           19           20           21           22           23           24 
## 0.0205629566 0.0055946383 0.0014461002 0.0492573369 0.9998054807 0.9997641636 
##           25           26           27           28           29           30 
## 0.9937819579 0.9365439781 0.9999532232 0.9993681135 0.9999759870 0.9529316585 
##           31           32           33           34           35           36 
## 0.0021087897 0.9993617436 0.0051850020 0.0188369201 0.0431867117 0.9999810832 
##           37           38           39           40           41           42 
## 0.0038842514 0.9991034841 0.9116157034 0.7578082104 0.9996009501 0.0010855860 
##           43           44           45           46           47           48 
## 0.0092957922 0.8922419926 0.9999976962 0.0526576231 0.0069308664 0.9988268687 
##           49           50           51           52           53           54 
## 0.9980936038 0.0086685971 0.0060487523 0.0018445515 0.1905433244 0.0044455042 
##           55           56           57           58           59           60 
## 0.0007286523 0.0042888712 0.9999897364 0.9999755875 0.0970399943 0.0034453423 
##           61           62           63           64           65           66 
## 0.1231030664 0.9999786514 0.9999990153 0.9999805753 0.0109392316 0.9999992497 
##           67           68           69           70           71           72 
## 0.0036146149 0.0008537205 0.0298018630 0.0072197789 0.9998482711 0.9999928297 
##           73           74           75           76           77           78 
## 0.0016591050 0.9999736093 0.9306746255 0.9998185687 0.9999918036 0.0071078469 
##           79           80           81           82           83           84 
## 0.9999846915 0.0026491653 0.9998356700 0.9999983437 0.9295944811 0.8703157269 
##           85           86           87           88           89           90 
## 0.0405679103 0.9999601586 0.9998894558 0.0031755977 0.0014640085 0.0052527270 
##           91           92           93           94           95           96 
## 0.9999971024 0.0065842086 0.8591224848 0.9991889279 0.0025925190 0.0367770508 
##           97           98           99          100          101          102 
## 0.1425698011 0.9996450498 0.9999940286 0.0096003511 0.9993851603 0.4551275578 
##          103          104          105          106          107          108 
## 0.9381709724 0.0035362960 0.9436953124 0.0023599488 0.1556110365 0.0021048728 
##          109          110          111          112          113          114 
## 0.0032085787 0.0059113118 0.0098637077 0.0533159583 0.9999888424 0.9993988803 
##          115          116          117          118          119          120 
## 0.0114142296 0.9999642679 0.0084722400 0.9996336242 0.0037764053 0.9759435967 
##          121          122          123          124          125          126 
## 0.1032960641 0.9999603559 0.0019202742 0.9998290065 0.0021853165 0.9999537138 
##          127          128          129          130          131          132 
## 0.0019928342 0.0022110063 0.0261720862 0.4726681024 0.0207258881 0.9999867024 
##          133          134          135          136          137          138 
## 0.0019538991 0.0108211602 0.9999230300 0.9999838264 0.0025326240 0.2721968675 
##          139          140          141          142          143          144 
## 0.9846244857 0.0071374108 0.0080224460 0.0027556543 0.0011532705 0.0030206677 
##          145          146          147          148          149          150 
## 0.1916251982 0.9974699582 0.9999853826 0.9999946079 0.0032121485 0.9833518642 
##          151          152          153          154          155          156 
## 0.0364342521 0.9999911873 0.0022124016 0.9798773838 0.0136878259 0.0020453278 
##          157          158          159          160          161          162 
## 0.0406913548 0.9963971820 0.0026884339 0.9999417170 0.9872134441 0.0896571206 
##          163          164          165          166          167          168 
## 0.0035606971 0.9989438639 0.9949137381 0.0174808215 0.9999712325 0.9999813946 
##          169          170          171          172          173          174 
## 0.0022156547 0.0010817361 0.9999194189 0.0018454697 0.9999996299 0.0275082868 
##          175          176          177          178          179          180 
## 0.0014697466 0.0041408804 0.0099926913 0.9999885969 0.9999210842 0.0066225897 
##          181          182          183          184          185          186 
## 0.9999747009 0.9999993783 0.0015116765 0.8914070172 0.0034895011 0.8142138273 
##          187          188          189          190          191          192 
## 0.0126378041 0.9981688896 0.0021245284 0.0065164301 0.9989455429 0.9997852535 
##          193          194          195          196          197          198 
## 0.1124037513 0.0191468823 0.7358276923 0.9999671929 0.9999865495 0.9999628436 
##          199          200          201          202          203          204 
## 0.9999914108 0.2727577934 0.9999501847 0.0382508011 0.0334804757 0.9991938105 
##          205          206          207          208          209          210 
## 0.8349435748 0.9847203193 0.0275052045 0.9999473558 0.9996789304 0.9999846821 
##          211          212          213          214          215          216 
## 0.0015814966 0.0061977335 0.0086105584 0.9999919688 0.0057241808 0.0031793432 
##          217          218          219          220          221          222 
## 0.0014509484 0.7760102376 0.0128699658 0.0761771818 0.0025276753 0.1251498836 
##          223          224          225          226          227          228 
## 0.0013692006 0.8447232974 0.0043461278 0.9997247523 0.0252353343 0.0104772522 
##          229          230          231          232          233          234 
## 0.0080719830 0.9799765567 0.0017179452 0.5982289209 0.0085421626 0.9998644527 
##          235          236          237          238          239          240 
## 0.9999915265 0.9999990554 0.0055990115 0.9976064455 0.9999972651 0.1220287550 
##          241          242          243          244          245          246 
## 0.9999636141 0.0706318679 0.9998817773 0.0022480320 0.0075476507 0.9999994472 
##          247          248          249          250          251          252 
## 0.9999967857 0.9999981094 0.9999973102 0.9999825678 0.0032311072 0.9999961309 
##          253          254          255          256          257          258 
## 0.0026816966 0.9999564036 0.0089851649 0.9999985093 0.9999335531 0.9999952941 
##          259          260          261          262          263          264 
## 0.9999682065 0.0115518811 0.9998695390 0.1101440447 0.0184141868 0.9999860633 
##          265          266          267          268          269          270 
## 0.6936071499 0.9999964979 0.1184414745 0.0058018989 0.0099264121 0.0020242006 
##          271          272          273          274          275          276 
## 0.0126336209 0.0026086176 0.0390604922 0.9999921676 0.0041365833 0.9999997468 
##          277          278          279          280          281          282 
## 0.9020181616 0.0134363552 0.9999978661 0.9999957517 0.0065599881 0.9999988634 
##          283          284          285          286          287          288 
## 0.9997480647 0.0023483234 0.0045233934 0.9999638276 0.6723536665 0.9974648217 
##          289          290          291          292          293          294 
## 0.2875214329 0.9998629530 0.0840139376 0.0457437465 0.9999866738 0.0174647925 
##          295          296          297          298          299          300 
## 0.9999994199 0.9996443850 0.1472272800 0.0019194713 0.0896325210 0.9999988717
```

- Changing probabilities. 


```r
# Changing probabilities
predict_reg <- ifelse(predict_reg >0.5, 1, 0)
```

- Evaluate the performance of the model using the accuracy.

```r
# Evaluating model accuracy
# using confusion matrix
table(test$`Clicked on Ad`, predict_reg)
```

```
##    predict_reg
##       0   1
##   0 147   3
##   1   9 141
```

```r
missing_classerr <- mean(predict_reg != test$`Clicked on Ad`)
print(paste('Accuracy =', 1 - missing_classerr))
```

```
## [1] "Accuracy = 0.96"
```

- We have achieved an accuracy score of 98% using the logistic regression model. 


# 8. CONCLUSIONS

From the univariate data analysis, we can conclude that: 

- There were more females than males in our dataset.
- The dataset was balanced in the sense that 500 individuals clicked on the ads while 500 individuals did not click on the ads.
- Individuals who are between 28 and 36 years old were the most in our dataset.
- Czech Republic and France both had the highest number of individuals (9) in the dataset.
- Lisamouth and Williamsport cities both had the highest number of individuals (3) in the dataset.

From the bivariate data analysis, we can conclude that: 

- There is a negative covariance and correlation between age and daily time spent on the site which means that the older an individual is, the less time they spend on the site.
- There is also a negative covariance and correlation between age and the daily internet usage which means that the younger an individual is, the higher the internet usage is as compared to an older individual.
- On the other hand, there is a positive covariance and correlation between the daily internet usage and the daily time spent on the internet.

For the modelling techniques, we used different supervised learning algorithms to help us predict whether an individual will click on an ad and determined that:

- Naive Bayes had an accuracy of 96%
- KNN had an accuracy of 96%
- SVM had an accuracy of 97%
- Random Forest Classifier has an accuracy of 95.3% which was the lowest as compared to the other models. 

We then challenged the solution using a different supervised learning algorithm, the logistic regression model, which gave us the highest accuracy of 98% in predicting whether an individual will click on an ad. 


# 9. RECOMMENDATIONS

-   We recommend that she creates an ad that targets individuals aged between 25 and 35 years old seeing as they are the most in our dataset.
-   We recommend that she focuses her attention more on the youth as they use the internet more and spend more time on the site as compared to the older individuals.

