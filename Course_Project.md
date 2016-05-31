*Abstract*
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

*Analysis*
----------

### Libraries

``` r
library(caret); library(ggplot2); library(dplyr)
```

### Download and Read Data

``` r
training.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download.file(url = training.url, destfile = "training.csv", method = "curl")
# download.file(url = test.url, destfile = "testing.csv", method = "curl")

training.f <- read.csv("training.csv")
testing.f <- read.csv("testing.csv")
dim(training.f); dim(testing.f)
```

    ## [1] 19622   160

    ## [1]  20 160

### Remove zero covariates

Zero covariates are poor predictors because of their low variability, so we remove them from the original dataset to reduce the size of the data.

``` r
zero.cov.index <- nearZeroVar(subset(training.f, select=-classe))
training.processed <- training.f[,-zero.cov.index]
dim(training.processed)
```

    ## [1] 19622   100

### Remove variables with more than 80% of NAs

Variables with more than 80% of NAs can hardly be good predictors, so we remove them from the original dataset to reduce the size of the data.

``` r
na.prop <- apply(is.na(training.processed), 2, function(x){sum(x)/nrow(training.processed)})
na.index <- unname(which(na.prop > 0.8))
training.processed1 <- training.processed[, -na.index]
dim(training.processed1)
```

    ## [1] 19622    59

### Manually unselect some variables

We don't need a user name or a specific time for generalized predictions, so we remove related variables from the data set.

``` r
training.processed2 <- training.processed1 %>%
        select(-(X:num_window))
```

### Create training and testing data sets

``` r
inTrain <- createDataPartition(training.processed2$classe, p=0.7, list=FALSE)
training <- training.processed2[inTrain, ]
testing <- training.processed2[-inTrain, ]
dim(training); dim(testing)
```

    ## [1] 13737    53

    ## [1] 5885   53

### Apply Random Forest Model

We choose to apply random forest model on the preprocessed data. We can see that the model has a very low out of sample error, which is around 1%.

``` r
mod.fit1 <- train(classe ~ ., data=training, method="rf",
                  trControl=trainControl(method="cv", number=3))
mod.fit1
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 9158, 9158, 9158 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9866055  0.9830533
    ##   27    0.9879158  0.9847130
    ##   52    0.9837665  0.9794649
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

### Check the accuracy on testing data

``` r
pred1 <- predict(mod.fit1, testing)
confusionMatrix(pred1, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    8    0    0    0
    ##          B    0 1129    8    0    0
    ##          C    0    2 1012   13    1
    ##          D    0    0    6  949    3
    ##          E    2    0    0    2 1078
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9924          
    ##                  95% CI : (0.9898, 0.9944)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9903          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9912   0.9864   0.9844   0.9963
    ## Specificity            0.9981   0.9983   0.9967   0.9982   0.9992
    ## Pos Pred Value         0.9952   0.9930   0.9844   0.9906   0.9963
    ## Neg Pred Value         0.9995   0.9979   0.9971   0.9970   0.9992
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1918   0.1720   0.1613   0.1832
    ## Detection Prevalence   0.2855   0.1932   0.1747   0.1628   0.1839
    ## Balanced Accuracy      0.9985   0.9948   0.9915   0.9913   0.9977

### Quiz (20/20)

``` r
testing.processed2 <- select(testing.f, -zero.cov.index) %>%
        select(-na.index) %>%
        select(-(X:num_window))
predict(mod.fit1, testing.processed2)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
