library(softImpute)
mat<-read.table('matrix.txt',sep='\t',header=T)
mat_scaled_t<-t(mat)
#mat_scaled_t<-scale(t(mat))
lam0<-lambda0(mat_scaled_t)
lamseq=exp(seq(from=log(lam0),to=log(1e-10),length=10))
fits=as.list(lamseq)
ranks=as.integer(lamseq)
rank.max<-1
#breaks<-dim(mat_scaled_t)[1]
breaks<-1
examples<-dim(mat_scaled_t)[2]
warm=NULL
for(i in seq(along=lamseq)){
        fiti<-softImpute(mat_scaled_t,trace.it=F,rank.max=rank.max,lambda=lamseq[i],warm=warm)
        ranks[i]<-sum(round(fiti$d,4)>0)
        rank.max<-min(ranks[i]+1,breaks)
        warm<-fiti
        fits[[i]]=fiti
        new_mat<-complete(mat_scaled_t,fiti)
        cat(i,"lambda=",lamseq[i],"rank.max",rank.max,"rank",ranks[i],"\n")
        write.table(t(new_mat),paste("imputed.rank",ranks[i],"lambda",lamseq[i],sep="."),quote=F)
        #mat_scale<-attributes(mat_scaled_t)$`scaled:scale`
        #mat_center<-attributes(mat_scaled_t)$`scaled:center`
        #new_mat2<-t(t(new_mat)*mat_scale+mat_center)
        #write.table(t(new_mat2),"imputed.txt",quote=F)
        #tests<-3
        tests<-5487
        #negative<-0
        #for(i in 1:(dim(mat_scaled_t)[1]-1)){
            #m<-matrix(new_mat[i,(examples-tests):examples]-new_mat[i+1,(examples-tests):examples])
            #negative<-negative+sum(m<0)
        #}
        cat("Negatives:",sum(new_mat<0),"\n",sep=" ")
}
        
