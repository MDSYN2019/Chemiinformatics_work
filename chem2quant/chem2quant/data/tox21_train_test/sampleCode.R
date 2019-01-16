#IF PACKAGES ARE MISSING RUN:
#install.packages(c("randomForest","Matrix","ROCR"))
library(randomForest)
library(Matrix)
library(ROCR)

AUC <- function(x,l){
        library(ROCR)
        naIdx <- is.na(x) | is.na(l)
        x <- x[!naIdx]
        l <- l[!naIdx]
        return(performance(prediction(x,l),"auc")@y.values[[1]][1])
}


### Load data
X.dense.train <- read.table("tox21_dense_train.csv.gz",sep=",",header=TRUE,row.names=1,check.names=FALSE)
X.dense.test <- read.table("tox21_dense_test.csv.gz",sep=",",header=TRUE,row.names=1,check.names=FALSE)

X.sparse.train = readMM('tox21_sparse_train.mtx.gz')
X.sparse.train=as(X.sparse.train, "dgCMatrix")
rownames(X.sparse.train)=readLines("tox21_sparse_rownames_train.txt.gz")
colnames(X.sparse.train)=readLines("tox21_sparse_colnames.txt.gz")

X.sparse.test = readMM('tox21_sparse_test.mtx.gz')
X.sparse.test=as(X.sparse.test, "dgCMatrix")
rownames(X.sparse.test)=readLines("tox21_sparse_rownames_test.txt.gz")
colnames(X.sparse.test)=readLines("tox21_sparse_colnames.txt.gz")

Y.train <- read.table("tox21_labels_train.csv.gz",sep=",",header=TRUE,row.names=1,check.names=FALSE)
Y.test <- read.table("tox21_labels_test.csv.gz",sep=",",header=TRUE,row.names=1,check.names=FALSE)

### Feature selection unsupervised
idxSelected <- which(colSums(X.sparse.train > 0) > 0.05*nrow(X.sparse.train))
X.train <- cbind(as.matrix(X.dense.train),as.matrix(X.sparse.train[,idxSelected]))
X.test <- cbind(as.matrix(X.dense.test),as.matrix(X.sparse.test[,idxSelected]))


### Build RandomForest model for NR.Ahr
idxTrain <- which(!is.na(Y.train[,1]))
rF <- randomForest(x=X.train[idxTrain, ],y=factor(Y.train[idxTrain,1]),ntree=100)

predictions <- predict(rF,X.test,type="prob")
cat("Prediction area under ROC curve:", AUC(predictions[,2],Y.test[,1]),"\n")


### Build RandomForest model for all twelve assays
for (i in 1:12){
        idxTrain <- which(!is.na(Y.train[,i]))
        rF <- randomForest(x=X.train[idxTrain, ],y=factor(Y.train[idxTrain,i]),ntree=100)

        predictions <- predict(rF,X.test,type="prob")
        cat(colnames(Y.train)[i]," -- prediction area under ROC curve:", AUC(predictions[,2],Y.test[,i]),"\n")
}
