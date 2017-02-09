library(MASS)
set.seed(1)
rows<-20
cols<-5
X<-matrix(,rows,cols)
for(i in 1:rows){
    X[i,]<-
    if(i>=1 & i<=10){
        X[i,]<-mvrnorm(,mu=c(5,4,3,2,1),Sigma=diag(cols)*.1)
    }else if(i>=11 & i<=20){
        #X[i,]<-mvrnorm(,mu=c(1,2,3,4,5),Sigma=diag(cols)*.01)
        X[i,]<-mvrnorm(,mu=c(1.5,1.4,1.3,1.2,1),Sigma=diag(cols)*.1)
    }else if(i>=8 & i<=15){
        X[i,]<-mvrnorm(,mu=c(1,1,1,1,1),Sigma=diag(cols)*.001)
    }
#    for(j in 1:cols){
#    }
}
cat("True matrix\n")
print(X)

Y<-X
Y[6:15,1:3]<-NA
cat("Matrix with unobserved\n")
Y2<-replace(Y,is.na(Y),-1)
print(Y)
write.table(Y2,file='gpu_input.txt',quote=F,row.names=F,col.names=F,sep='\t')

library(softImpute)
source('nmf.r')
for(rank in 1:1){
#for(rank in 1:(cols-1)){
    cat("Rank",rank,"\n")
    #fiti<-softImpute(Y,trace.it=F,rank.max=rank)
    #imputed<-complete(Y,fiti)
    #print(imputed)
    imputed<-impute(Y,rank)
    cat("NMF\n")
    print(imputed)
    #cat("\n")
    cat("L2 norm of error:",norm(imputed-X,type='f'),"\n")
}
