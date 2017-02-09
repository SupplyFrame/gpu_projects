source('nmf.r')
dat<-read.table('matrix.txt',sep='\t',header=T)
mat<-as.matrix(dat)
mat<-replace(mat,which(mat==0),1e-10)
for(rank in 1:5){
    cat("Computing on rank",rank,"\n")
    #rank<-1
    imputed<-impute(mat,rank)
    df<-as.data.frame(imputed)
    rownames(df)<-rownames(dat)
    colnames(df)<-colnames(dat)
    write.table(df,file=paste('imputed_nnmf_rank',rank,'.txt',sep=''),quote=F)
}
