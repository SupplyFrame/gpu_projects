
nnmf<-function(V,rank){
    rows<-dim(V)[1]
    cols<-dim(V)[2]
    set.seed(1)
    W<-matrix((runif(rows*rank)),rows,rank)
    H<-matrix((runif(rank*cols)),rank,cols)
    WH<-matrix(0,rows,cols)
    WH_last<-WH
    for(i in 1:1000){
        H_num<-t(W)%*%V
        H_den<-t(W)%*%W%*%H
        H_update<-H_num/H_den
        H<-H*H_update
        W_num<-V%*%t(H)
        W_den<-W%*%H%*%t(H)
        W_update<-W_num/W_den
        W<-W*W_update
        WH<-W%*%H
        #cat(i,"\n")
        #cat("W\n")
        #cat(W)
        #cat("\n")
        #cat("H\n")
        #cat(H)
        #cat("\n")
        #cat("WH\n")
        #cat(WH)
        #cat("\n")
        diff<-norm(WH-WH_last,type='f')
        WH_last<-WH
        #cat(i,"Inner Diff",diff,"\n")
        if(diff<.001) break;
    }
    WH
}

impute<-function(observed,rank){
    rows<-dim(observed)[1]
    cols<-dim(observed)[2]
    observed_vec<-as.vector(observed)
    
    imputed_vec<-replace(observed_vec,is.na(observed_vec),0)
    imputed<-matrix(imputed_vec,rows,cols)
    last_imputed<-imputed

    for(r in rank:rank){
        for(i in 1:100000){
            #cat(paste("Iteration",i,"\n"))
            imputed<-nnmf(imputed,r)
            imputed_vec<-replace(observed_vec,which(is.na(observed_vec)),as.vector(imputed)[which(is.na(observed_vec))])
            imputed<-matrix(imputed_vec,rows,cols)
            #cat("predicted\n")
            #cat(imputed)
            #cat("\n")
            diff<-norm(imputed-last_imputed,type='f')
            #cat(paste("L2 norm of Outer diff to prev",diff,"\n"))
            if(diff<0.001) break;
            last_imputed<-imputed
        }
    }
    imputed
}

testing<-function(){
    rows<-7
    cols<-5
    rank<-2
    
    true<-matrix(1e-10,rows,cols)
    true[1,5]<-runif(1)
    true[2,4]<-runif(1)
    true[3,3]<-runif(1)
    cat(true)
    #true<-matrix(runif(rows*cols),rows,cols)
    observed<-true
    observed[1,1]<-NA
    observed[3,3]<-NA
    #cat(dim(observed))
    imputed<-impute(observed,rank)
    cat("true V\n")
    cat(true)
    cat("\n")
    diff2<-norm(imputed-true,type='f')
    cat(paste("L2 norm of diff to truth",diff2,"\n"))
}
