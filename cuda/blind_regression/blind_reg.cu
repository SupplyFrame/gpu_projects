#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<cuda_runtime.h>
#include<gsl/gsl_cblas.h>
#include<map>
#include"cublas_v2.h"
#define BLOCK_SIZE 256
#define idx(row,col) (col*n+row)

using namespace std;

    
struct matrix_t{
    int n;
    int m;
    float * data;
    bool * is_missing;
    //cublasHandle_t handle = NULL;
    //float* dev_ptr = NULL;
    matrix_t(const char * inputfile,int rows,int cols);
    void estimate_gaussian();
    void populate_intersecting_cols(int row,bool * intersecting_cols);
    void populate_intersecting_rows(int col,bool * intersecting_rows);
    void populate_best_row_neighbors(float beta,int row,int col,bool * intersecting_cols, bool * best_row_neighbors);
    void populate_best_col_neighbors(float beta,int col,int row,bool * intersecting_rows, bool * best_col_neighbors);
    void populate_best_neighbors(bool * best_row_neighbors, bool * best_col_neighbors, bool * best_neighbors);

    ~matrix_t(){
        if(data!=NULL) delete[] data;
        if(is_missing!=NULL) delete[] is_missing;
        //if(handle!=NULL) cublasDestroy(handle);
        //if(dev_ptr!=NULL) cudaFree(dev_ptr);
    }
};

#define M 6
#define N 5
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

cublasHandle_t handle = NULL;
cublasStatus_t stat;
typedef map<const char *,float *> dev_ptr_map_t;

dev_ptr_map_t dev_ptr_map;

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    printf("Calling scale %7.0f on number of elements %d on vector with index %d at coordinate %d %d and stride %d\n",alpha,n-p+1,IDX2F(p,q,ldm),p,q,ldm);
    cublasSscal (handle, n-p+1, &alpha, m+IDX2F(p,q,ldm), ldm); 
    //printf("Calling scale %7.0f on number of elements %d on vector with index %d at coordinate %d %d and stride %d\n",beta,ldm-p+1,IDX2F(p,q,ldm),p,q,1);
    //cublasSscal (handle, ldm-p+1, &beta, &m[IDX2F(p,q,ldm)], 1);
} 

__global__
void divide_matrix(int n,float * a,float * b, float *c){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) c[i] = c[i] * a[i]/b[i];
}

__global__
void diff_matrix(int n,float * a,float * b, float *c){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) c[i] = (a[i]-b[i])*(a[i]-b[i]);
}

__global__
void copy_matrix(int n,float * a,float * b){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) b[i] = a[i];
}

__global__
void reduction(int n,float * input, float *output){
    __shared__ float s_data[BLOCK_SIZE];
    uint tid = threadIdx.x;
    uint i = blockIdx.x*blockDim.x + tid;
    s_data[tid] = 0;
    if(i<n){
        //s_data[tid] = 1;
        s_data[tid] = input[i];
    }
    __syncthreads();
    if(tid<128){s_data[tid]+=s_data[tid+128];} __syncthreads();
    if(tid<64){s_data[tid]+=s_data[tid+64];} __syncthreads();
    if(tid<32){s_data[tid]+=s_data[tid+32];} __syncthreads();
    if(tid<16){s_data[tid]+=s_data[tid+16];} __syncthreads();
    if(tid<8){s_data[tid]+=s_data[tid+8];} __syncthreads();
    if(tid<4){s_data[tid]+=s_data[tid+4];} __syncthreads();
    if(tid<2){s_data[tid]+=s_data[tid+2];}__syncthreads();
    if(tid<1){s_data[tid]+=s_data[tid+1];}__syncthreads();
        //s_data[tid]+=s_data[tid+8];
        //s_data[tid]+=s_data[tid+4];
        //s_data[tid]+=s_data[tid+2];
        //s_data[tid]+=s_data[tid+1];
    //}
    if(tid==0) output[blockIdx.x] = s_data[0];
}


float l2_norm(float * imputed,float * last_imputed,int n){
    float norm = 0;
    for(int i =0;i<n;++i){
        norm+=(imputed[i]-last_imputed[i])*(imputed[i]-last_imputed[i]);
    }
    return sqrt(norm);
}

inline float runif(){
    return rand()/(RAND_MAX+1.0);

}

void debug_matrix(const char *label,float * mat,int rows,int cols){
    printf("%s\n",label);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            if(j) printf("\t");
            printf("%f",mat[j*rows+i]);
        }
        printf("\n");
    }
}

inline void push_matrix(int rows,int cols,float * cpu, float * gpu){
    stat = cublasSetMatrix(rows,cols,sizeof(float),cpu,rows,gpu,rows);
}

inline void fetch_matrix(int rows,int cols,float * gpu, float * cpu){
    stat = cublasGetMatrix(rows,cols,sizeof(float),gpu,rows,cpu,rows);
    if(stat!=CUBLAS_STATUS_SUCCESS){
        printf("data upload failed.\n");
    }
}

float * get_dev_ptr(const char * label,int rows,int cols,float * orig){
//float * get_dev_ptr(int rows,int cols,float * orig){
    if(dev_ptr_map.find(label)!=dev_ptr_map.end()){
        printf("Using cached dev_ptr of label %s\n",label);
        return dev_ptr_map[label];
    }
    cudaError_t cuda_error;
    cublasStatus_t stat;
    float * dev_ptr = NULL;
    cuda_error = cudaMalloc ((void**)&dev_ptr, rows*cols*sizeof(float));
    if(cuda_error!=cudaSuccess){
	printf("device memory allocation failed.");
	cudaFree(dev_ptr);
        exit(1);
    }else{
        //printf("Allocated!\n");
    }
    if(orig!=NULL){
        push_matrix(rows,cols,orig,dev_ptr);
        //debug_matrix("before orig",orig,rows,cols);
        stat = cublasGetMatrix(rows,cols,sizeof(float),dev_ptr,rows,orig,rows);
        if(stat!=CUBLAS_STATUS_SUCCESS){
            printf("data upload failed.\n");
        }else{
            //debug_matrix("after orig",orig,rows,cols);
        }
    }
    printf("Storing new dev ptr of label %s\n",label);
    dev_ptr_map[label] = dev_ptr;
    return dev_ptr;
}

void nnmf(float * imputed,int rows,int cols,int rank){

    bool demo = false;
    if(demo){
    
      int lda = 3;
      float A[] = { 0.11, 0.12, 0.13,
                    0.21, 0.22, 0.23 };

      int ldb = 2;
  
      float B[] = { 1011, 1012,
                1021, 1022,
                    1031, 1032 };

      int ldc = 2;

      float D[] = { 0.00, 0.00,0.0,
                0.00, 0.00,0.0,
                    0.00, 0.00,0.0 };
  float C[] = { 0.00, 0.00,
                    0.00, 0.00 };

      /* Compute C = A B */

      cblas_sgemm (CblasRowMajor, 
               CblasTrans, CblasTrans, 3, 3, 2,
                   1.0, A, lda, B, ldb, 0.0, D, 3);

      printf ("[ %g, %g %g\n", D[0], D[1],D[2]);
        printf ("  %g, %g %g \n", D[3], D[4],D[5]);
          printf ("  %g, %g %g ]\n", D[6], D[7],D[8]);

      cblas_sgemm (CblasRowMajor, 
                   CblasNoTrans, CblasNoTrans, 2, 2, 3,
               1.0, A, lda, B, ldb, 0.0, C, ldc);

      printf ("[ %g, %g\n", C[0], C[1]);
      printf ("  %g, %g ]\n", C[2], C[3]);
    }


    float w[rows*rank];
    float h[rank*cols];
    float wh[rows*cols];
    float wh_last[rows*cols];
    memset(wh_last,0,sizeof(wh_last));

    for(int i=0;i<rows*rank;++i) w[i] = runif();
    for(int i=0;i<rank*cols;++i) h[i] = runif();
        //debug_matrix("w",w,rows,rank);
        //debug_matrix("h",h,rank,cols);
    
    //for(int i=0;i<rows*cols;++i) wh[i] = 0;
    int maxiter = 10;
    for(int iter = 0;iter<maxiter;++iter){
        float alpha = 1.0;
        float beta = 0.0;

        float h_num[rank*cols];
        memset(h_num,0,sizeof(h_num));
        cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,rank,cols,rows,alpha,w,rows,imputed,rows,beta,h_num,rank);
        debug_matrix("h_num_cpu",h_num,rank,cols);

        memset(h_num,0,sizeof(h_num));
        float * dev_w  = get_dev_ptr("w",rows,rank,w);
        float * dev_imputed = get_dev_ptr("imputed",rows,cols,imputed);
        float * dev_h_num = get_dev_ptr("h_num",rank,cols,h_num);
        stat = cublasSgemm(handle, CUBLAS_OP_T,CUBLAS_OP_N,rank,cols,rows,&alpha,dev_w,rows,dev_imputed,rows,&beta,dev_h_num,rank);
        fetch_matrix(rank,cols,dev_h_num,h_num);
        debug_matrix("h_num_gpu",h_num,rank,cols);

        float wtw[rank*rank];
        memset(wtw,0,sizeof(wtw));
        cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,rank,rank,rows,1.0,w,rows,w,rows,0.0,wtw,rank);
        debug_matrix("wtw_cpu",wtw,rank,rank);

        float * dev_wtw = get_dev_ptr("wtw",rank,rank,wtw);
        stat = cublasSgemm(handle, CUBLAS_OP_T,CUBLAS_OP_N,rank,rank,rows,&alpha,dev_w,rows,dev_w,rows,&beta,dev_wtw,rank);
        fetch_matrix(rank,rank,dev_wtw,wtw);
        debug_matrix("wtw_gpu",wtw,rank,rank);

        float h_den[rank*cols];
        memset(h_den,0,sizeof(h_den));
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,rank,cols,rank,1.0,wtw,rank,h,rank,0.0,h_den,rank);
        debug_matrix("h_den_cpu",h_den,rank,cols);

        float * dev_h = get_dev_ptr("h",rank,cols,h);
        float * dev_h_den = get_dev_ptr("h_den",rank,cols,h_den);
        stat = cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N,rank,cols,rank,&alpha,dev_wtw,rank,dev_h,rank,&beta,dev_h_den,rank);
        fetch_matrix(rank,cols,dev_h_den,h_den);
        debug_matrix("h_den_gpu",h_den,rank,cols);

        for(int i=0;i<rank*cols;++i) h[i] *= h_num[i]/h_den[i];
        debug_matrix("h_cpu",h,rank,cols);

        //memset(h,0,sizeof(h));
        //push_matrix(rank,cols,h,dev_h);
        divide_matrix<<<(rank*cols+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE >>>(rank*cols,dev_h_num,dev_h_den,dev_h);
        fetch_matrix(rank,cols,dev_h,h);
        debug_matrix("h_gpu",h,rank,cols);

        float w_num[rows*rank];
        memset(w_num,0,sizeof(w_num));
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,rows,rank,cols,1.0,imputed,rows,h,rank,0.0,w_num,rows);
        debug_matrix("w_num_cpu",w_num,rows,rank);

        memset(w_num,0,sizeof(w_num));
        //float * dev_imputed = get_dev_ptr("imputed",rows,cols,imputed);
        //float * dev_h  = get_dev_ptr("h",rank,cols,h);
        float * dev_w_num = get_dev_ptr("w_num",rows,rank,w_num);
        stat = cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T,rows,rank,cols,&alpha,dev_imputed,rows,dev_h,rank,&beta,dev_w_num,rows);
        fetch_matrix(rows,rank,dev_w_num,w_num);
        debug_matrix("w_num_gpu",w_num,rows,rank);

        float hht[rank*rank];
        memset(hht,0,sizeof(hht));
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,rank,rank,cols,1.0,h,rank,h,rank,0.0,hht,rank);
        debug_matrix("hht_cpu",hht,rank,rank);

        memset(hht,0,sizeof(hht));
        float * dev_hht  = get_dev_ptr("hht",rank,rank,hht);
        stat = cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T,rank,rank,cols,&alpha,dev_h,rank,dev_h,rank,&beta,dev_hht,rank);
        fetch_matrix(rank,rank,dev_hht,hht);
        debug_matrix("hht_gpu",hht,rank,rank);

        float w_den[rows*rank];
        memset(w_den,0,sizeof(w_den));
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,rows,rank,rank,1.0,w,rows,hht,rank,0.0,w_den,rows);
        debug_matrix("w_den_cpu",w_den,rows,rank);

        memset(w_den,0,sizeof(w_den));
        float * dev_w_den  = get_dev_ptr("w_den",rows,rank,w_den);
        stat = cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N,rows,rank,rank,&alpha,dev_w,rows,dev_hht,rank,&beta,dev_w_den,rows);
        fetch_matrix(rows,rank,dev_w_den,w_den);
        debug_matrix("w_den_gpu",w_den,rows,rank);

        for(int i=0;i<rows*rank;++i) w[i] *= w_num[i]/w_den[i];
        debug_matrix("w_cpu",w,rows,rank);
        //push_matrix(rows,rank,w,dev_w);
        divide_matrix<<<(rows*rank+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE >>>(rows*rank,dev_w_num,dev_w_den,dev_w);
        fetch_matrix(rows,rank,dev_w,w);
        debug_matrix("w_gpu",w,rows,rank);

        memset(wh,0,sizeof(wh));
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,rows,cols,rank,1.0,w,rows,h,rank,0.0,wh,rows);
        debug_matrix("wh_cpu",wh,rows,cols);

        memset(wh,0,sizeof(wh));
        float * dev_wh  = get_dev_ptr("wh",rows,cols,wh);
        stat = cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N,rows,cols,rank,&alpha,dev_w,rows,dev_h,rank,&beta,dev_wh,rows);

        fetch_matrix(rows,cols,dev_wh,wh);
        debug_matrix("wh_gpu",wh,rows,cols);

        float norm = l2_norm(wh,wh_last,rows*cols);
        printf("l2 norm cpu %f\n",norm);
        float * dev_wh_last  = get_dev_ptr("wh_last",rows,cols,wh_last);
        float wh_diff[rows*cols];
        memset(wh_diff,0,sizeof(wh_diff));
        float * dev_wh_diff  = get_dev_ptr("wh_diff",rows,cols,wh_diff);
        float wh_norm[rows*cols];
        memset(wh_norm,0,sizeof(wh_norm));
        float * dev_wh_norm  = get_dev_ptr("wh_norm",rows,cols,wh_norm);
        int n = rows*cols;
        int thread_blocks = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
        printf("For n %d, Launching %d thread blocks\n",n,thread_blocks);
        diff_matrix<<<thread_blocks,BLOCK_SIZE >>>(n,dev_wh_last,dev_wh,dev_wh_diff);
        fetch_matrix(rows,cols,dev_wh_diff,wh_diff);
        float gpu_norm = 0;
        for(int j=0;j<rows*cols;++j){
            gpu_norm+=wh_diff[j];
        }
        gpu_norm = sqrt(gpu_norm);
        printf("l2 norm gpu 1 %f\n",gpu_norm);
        float * dev_input = dev_wh_diff;
        float * dev_output = dev_wh_norm;
        while(thread_blocks){
            printf("Now launching %d thread blocks with n %d\n",thread_blocks,n);
            reduction<<<thread_blocks,BLOCK_SIZE >>>(n,dev_input,dev_output);
            n = thread_blocks>1?thread_blocks:0;
            thread_blocks = n?(n+BLOCK_SIZE-1)/BLOCK_SIZE:0;
            if(thread_blocks){
                printf("Swapping!\n");
                float * temp = dev_input;
                dev_input = dev_output;
                dev_output = temp;
            }
        }
        fetch_matrix(rows,cols,dev_output,wh_norm);
        wh_norm[0] = sqrt(wh_norm[0]);
        debug_matrix("l2 norm gpu", wh_norm,1,1);
        //printf("l2 norm gpu %f\n",norm);
        printf("Iteration %d: Inner L2 norm of diff %f\n",iter,norm);
        for(int i=0;i<rows*cols;++i) wh_last[i] = wh[i];
        n = rows*cols;
        thread_blocks = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
        copy_matrix<<<thread_blocks,BLOCK_SIZE >>>(n,dev_wh,dev_wh_last);
        exit(0);
    }
    for(int i=0;i<rows*cols;++i) imputed[i] = wh_last[i] ;
}

void impute(matrix_t * matrix,int rank){
    srand(1);
    int n = matrix->n;
    int m = matrix->m;
    bool * is_missing = matrix->is_missing;
    float * imputed = matrix->data;
    float * original = new float[n*m];
    float * last_imputed = new float[n*m];
    for(int i=0;i<n*m;++i){
        if(is_missing[i]) imputed[i] = 0;
        last_imputed[i] = original[i] = imputed[i];
    }
    int max_iter=100;
    for(int iter=0;iter<max_iter;++iter){
        nnmf(imputed,n,m,rank);
        for(int i=0;i<n*m;++i){
            if(!is_missing[i]) imputed[i] = original[i];
        }
        float diff = l2_norm(imputed,last_imputed,n*m);
        printf("Iteration %d Outer L2 norm of diff %f\n",iter, diff);
        if(diff<.01) break;
        for(int i=0;i<n*m;++i){
            last_imputed[i] = imputed[i];
        }
    }
}

int gpu_demo(){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i,j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc(M*N*sizeof(*a));
    if(!a){
	printf("host memory allocation failed.");
	return EXIT_FAILURE;
    }
    for(j=1;j<=N;j++){
	for(i=1;i<=M;i++){
	    //printf("Assigning to index %d value %.2f\n",IDX2F(i,j,M),(float)((i-1)*M+j));
	    a[IDX2F(i,j,M)] = (float)((i-1)*M+j);
	}
    }
    for(int i=0;i<N;++i){
	for(int j=0;j<M;++j){
	    //printf(" %.2f",a[i*M+j]);
	}
	//printf("\n");
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if(cudaStat!=cudaSuccess){
	printf("device memory allocation failed.");
	return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if(stat!=CUBLAS_STATUS_SUCCESS){
	printf("CUBLAS initialization failed.\n");
	return EXIT_FAILURE;
    }
    stat = cublasSetMatrix(M,N,sizeof(*a),a,M,devPtrA,M);
    if(stat!=CUBLAS_STATUS_SUCCESS){
	printf("data download failed.\n");
	cudaFree(devPtrA);
	cublasDestroy(handle);
	return EXIT_FAILURE;

    }
    modify(handle,devPtrA,M,N,2,3,16.0f,12.0f);
    stat = cublasGetMatrix(M,N,sizeof(*a),devPtrA,M,a,M);
    if(stat!=CUBLAS_STATUS_SUCCESS){
	printf("data upload failed.\n");
	cudaFree(devPtrA);
	cublasDestroy(handle);
	return EXIT_FAILURE;
    }
    cudaFree(devPtrA);
    cublasDestroy(handle);
    for(j=1;j<=N;++j){
	for(i=1;i<=M;++i){
	    printf("%7.0f",a[IDX2F(i,j,M)]);
	}
	printf("\n");
    }
    free(a);
    return 0;

}

matrix_t::matrix_t(const char * inputfile,int rows,int cols){
    n = rows;
    m = cols;
    data = new float[rows*cols];
    is_missing = new bool[rows*cols];
    FILE * file_ptr = fopen(inputfile,"r");
    if(file_ptr==NULL) {
        printf("Cannot open %s\n",inputfile);
        exit(1);
    }
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            fscanf(file_ptr,"%f",data+(j*rows+i));
            is_missing[j*rows+i] = data[j*rows+i]<0?true:false;
        }
    }
    //debug_matrix("After read",matrix->data,rows,cols);
    fclose(file_ptr);
    cublasStatus_t cublas_stat = cublasCreate(&handle);
    if(cublas_stat!=CUBLAS_STATUS_SUCCESS){
	printf("CUBLAS initialization failed.\n");
        exit(1);
    }
}

void cleanup(){
    for(dev_ptr_map_t::iterator it = dev_ptr_map.begin();
    it!=dev_ptr_map.end();++it){
        cudaFree(it->second);
    }
    cublasDestroy(handle);
    printf("Freed all memory\n");
}

void matrix_t::populate_intersecting_cols(int row,bool * intersecting_cols){
    for(int i=0;i<n;++i){
        //printf("Populate intersecting on row %d anchor %d\n",i,row);
        for(int j=0;j<m;++j){
            if(data[idx(i,j)]>0 && data[idx(row,j)]>0){
                intersecting_cols[idx(i,j)] = true;
                //printf(" %d",j);
            }
        }
        //printf("\n");
    }
}

void matrix_t::populate_intersecting_rows(int col,bool * intersecting_rows){
    for(int j=0;j<m;++j){
        //printf("Populate intersecting on col %d anchor %d\n",j,col);
        for(int i=0;i<n;++i){
            if(data[idx(i,j)]>0 && data[idx(i,col)]>0){
                intersecting_rows[idx(i,j)] = true;
                //printf(" %d",i);
            }
        }
        //printf("\n");
    }
}

void matrix_t::populate_best_row_neighbors(float beta, int row,int col,bool * intersecting_cols, bool * best_row_neighbors){
    //printf("Beta row at col %d\n",col);
    for(int i=0;i<n;++i){
        if(data[idx(i,col)]>0 && i!=row){
            int support = 0;
            for(int j=0;j<m;++j){
                support+=intersecting_cols[idx(i,j)];
            }
            if(support>=beta){
                best_row_neighbors[i] = true;
                //printf(" %d",i);
            }
        }
    }
    //printf("\n");
}

void matrix_t::populate_best_col_neighbors(float beta, int col,int row,bool * intersecting_rows, bool * best_col_neighbors){
    //printf("Beta col at row %d\n",row);
    for(int j=0;j<m;++j){
        if(data[idx(row,j)]>0 && j!=col){
            int support = 0;
            for(int i=0;i<n;++i){
                support+=intersecting_rows[idx(i,j)];
            }
            if(support>=beta){
                best_col_neighbors[j] = true;
                //printf(" %d",j);
            }
        }
    }
    //printf("\n");
}

void matrix_t::populate_best_neighbors(bool * best_row_neighbors, bool * best_col_neighbors,bool * best_neighbors){
    printf("Best neighbors:");
    for(int j=0;j<m;++j){
        for(int i=0;i<n;++i){
            if(best_row_neighbors[i] && best_col_neighbors[j] && data[idx(i,j)]>0){
                best_neighbors[idx(i,j)] = true;
                printf(" %d,%d",i,j);
            }
        }
    }
    printf("\n");
}

void matrix_t::estimate_gaussian(){
    bool * intersecting_cols = new bool[m*n];
    bool * intersecting_rows = new bool[m*n];
    bool * best_row_neighbors = new bool[n];
    bool * best_col_neighbors = new bool[m];
    bool * best_neighbors = new bool[n*m];
    float beta = 5.0;
    for(int row=0;row<n;++row){
        memset(intersecting_cols,0,sizeof(bool)*m*n);
        populate_intersecting_cols(row, intersecting_cols);
        printf("At row %d\n",row);
        for(int col=0;col<m;++col){
            printf("At col %d\n",col);
            memset(best_row_neighbors,0,sizeof(bool)*n);
            populate_best_row_neighbors(beta, row, col, intersecting_cols, best_row_neighbors);
            memset(intersecting_rows,0,sizeof(bool)*m*n);
            populate_intersecting_rows(col, intersecting_rows);
            memset(best_col_neighbors,0,sizeof(bool)*m);
            populate_best_col_neighbors(beta, col,row, intersecting_rows, best_col_neighbors);
            memset(best_neighbors,0,sizeof(bool)*n*m);
            populate_best_neighbors(best_row_neighbors, best_col_neighbors,best_neighbors);
        }
    }
    delete[] best_neighbors;
    delete[] best_row_neighbors;
    delete[] best_col_neighbors;
    delete[] intersecting_cols;
    delete[] intersecting_rows;
//   for(int j=0;j<m;++j){
//      printf("At col %d\n",j);
//      for(int i=0;i<n;++i){
//         if(data[j*n+i]>0.0){
//            row_positive[i]=true;
//            col_positive[j]=true;
//            printf(" %d",i);
//         }
//      }
//      printf("\n");
//   }
//   printf("Row positive:");
//   for(int i=0;i<n;++i){
//      if(row_positive[i]) printf(" %d",i);
//   }
//   printf("\n");
//   printf("Col positive:");
//   for(int j=0;j<m;++j){
//      if(col_positive[j]) printf(" %d",j);
//   }
//   printf("\n");
         //if(j) printf(" ");
         //printf("%.2f",data[j*n+i]);
      //}
      //printf("\n");
   //}
   
	
}
    
int main(int argc,char * argv[]){
    if(argc<4){
	printf("usage: <matrix to impute> <rows> <cols>\n");
	return EXIT_FAILURE;
    }
    int arg = 0;
    const char * inputfile = argv[++arg];
    int n = atoi(argv[++arg]);
    int m = atoi(argv[++arg]);
    matrix_t * matrix = new matrix_t(inputfile,n,m);
    matrix->estimate_gaussian();
    //int rank = 1;
    //impute(matrix,rank);
    //cleanup();
    delete matrix;
    //gpu_demo();
    return(EXIT_SUCCESS);
}
