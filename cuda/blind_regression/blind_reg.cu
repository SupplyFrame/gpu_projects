#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<cuda_runtime.h>
#include<gsl/gsl_cblas.h>
#include<map>
#include<set>
#include<vector>
#include"cublas_v2.h"
#define BLOCK_SIZE 256
#define idx(row,col) (col*n+row)

using namespace std;

typedef pair<int,int> int_pair_t;
typedef vector<int> int_vec_t;
typedef set<int> int_set_t;

typedef vector<int_vec_t> int_vec_vec_t ;
typedef vector<int_vec_vec_t> int_vec_vec_vec_t ;
typedef vector<int_set_t> int_set_vec_t ;
typedef vector<int_pair_t> int_pair_vec_t;
typedef set<int_pair_t> int_pair_set_t;
    
struct matrix_t{
    float beta;
    int limit;
    int n;
    int m;
    float * data;
    float * data_rowmajor;
    float * data_colmajor;

    int_pair_set_t positive_indices;
    bool * positive_indices_mat;

    int_set_t ** positive_cols_indices2;
    int_set_t ** positive_rows_indices2;

    int_vec_t ** intersecting_cols_indices2;
    int_vec_t *** intersecting_rows_indices_by_anchor2;

    int_vec_t best_row_neighbors;
    //int_set_t best_col_neighbors;
    bool * best_col_neighbors;
    int_pair_vec_t best_neighbors;

    float * row_cache;
    float * col_cache;

    bool * row_cache_used;
    bool * col_cache_used;

    matrix_t(const char * inputfile,int rows,int cols);
    ~matrix_t();
    float * estimate_gaussian();
    void populate_intersecting_cols(int row);
    void populate_intersecting_rows(int col);
    void populate_best_row_neighbors(int row,int col);
    void populate_best_col_neighbors(int col,int row);
    void populate_best_neighbors(int col);
    float compute_square(int row,int col);
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
    beta = 5.0;
    limit = 10000;
    n = rows;
    m = cols;
    data = new float[rows*cols];
    best_col_neighbors = new bool[m];
    data_rowmajor = new float[rows*cols];
    data_colmajor = new float[cols*rows];
    positive_cols_indices2 = new int_set_t*[n];
    positive_indices_mat = new bool[n*m];
    memset(positive_indices_mat,0,sizeof(bool)*n*m);
    intersecting_cols_indices2 = new int_vec_t*[n];
    for(int i=0;i<n;++i){
        vector<int> v;
        set<int> s;
        intersecting_cols_indices2[i] = new vector<int>;
        positive_cols_indices2[i] = new set<int>;
    }
    positive_rows_indices2 = new int_set_t*[m];
    intersecting_rows_indices_by_anchor2 = new int_vec_t**[m];
    for(int j=0;j<m;++j){
        set<int> s;
        positive_rows_indices2[j] = new set<int>;
        vector<int> v;
        intersecting_rows_indices_by_anchor2[j] = new vector<int>*[m];
        for(int k=0;k<m;++k){
            intersecting_rows_indices_by_anchor2[j][k] = new vector<int>;
        }
    }
    row_cache = new float[n*n];
    row_cache_used = new bool[n*n];
    col_cache = new float[m*m];
    col_cache_used = new bool[n*n];

    FILE * file_ptr = fopen(inputfile,"r");
    if(file_ptr==NULL) {
        printf("Cannot open %s\n",inputfile);
        exit(1);
    }
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            float f;
            fscanf(file_ptr,"%f",&f);
            data[idx(i,j)] = f;
            data_rowmajor[i*cols+j] = f; 
            data_colmajor[j*rows+i] = f; 
            if(f>0){
                positive_indices.insert(make_pair(i,j));
                positive_cols_indices2[i]->insert(j);
                positive_rows_indices2[j]->insert(i);
                positive_indices_mat[idx(i,j)] = true;
            }
        }
    }
    fclose(file_ptr);
    cublasStatus_t cublas_stat = cublasCreate(&handle);
    if(cublas_stat!=CUBLAS_STATUS_SUCCESS){
	printf("CUBLAS initialization failed.\n");
        exit(1);
    }
}

matrix_t::~matrix_t(){
}

void cleanup(){
    for(dev_ptr_map_t::iterator it = dev_ptr_map.begin();
    it!=dev_ptr_map.end();++it){
        cudaFree(it->second);
    }
    cublasDestroy(handle);
    printf("Freed all memory\n");
}

void matrix_t::populate_intersecting_cols(int row){
    bool debug = false;
    if(debug) printf("Intersecting cols at row %d\n",row);
    for(int i=0;i<n;++i){
        if(debug)printf("Populate intersecting on row %d anchor %d\n",i,row);
        intersecting_cols_indices2[i]->clear();
        for(int_set_t::iterator it = positive_cols_indices2[i]->begin();
        it!=positive_cols_indices2[i]->end();++it){
            int col = *it;
            if(positive_cols_indices2[row]->find(col)!=positive_cols_indices2[row]->end()){
                if(debug)printf("Adding at col %d\n",col);
                intersecting_cols_indices2[i]->push_back(col);
            }
        }
    }
}

void matrix_t::populate_intersecting_rows(int col){
    bool debug = false;
    if(debug) printf("Intersecting rows at anchor col %d\n",col);
    for(int j=0;j<m;++j){
        if(debug)printf("Populate intersecting on col %d anchor %d\n",j,col);
        intersecting_rows_indices_by_anchor2[col][j]->clear();
        for(int_set_t::iterator it = positive_rows_indices2[j]->begin();
        it!=positive_rows_indices2[j]->end();++it){
            int row = *it;
            if(positive_rows_indices2[col]->find(row)!=positive_rows_indices2[col]->end()){
                intersecting_rows_indices_by_anchor2[col][j]->push_back(row);
            }
            if(debug)printf("Col %d is size %d\n",j,(int)intersecting_rows_indices_by_anchor2[col][j]->size());
        }
    }
}

void matrix_t::populate_best_row_neighbors(int anchor_row,int col){
    bool debug = false;
    if(debug) printf("At beta %.2f best row neighbors at col %d\n",beta,col);
    best_row_neighbors.clear();
    for(int_set_t::iterator it = positive_rows_indices2[col]->begin();
    it!=positive_rows_indices2[col]->end();++it){
        int row = *it;
        if(row!=anchor_row && intersecting_cols_indices2[row]->size()>=beta){
            if(debug)printf("Adding best row neighbor %d\n",row);
            best_row_neighbors.push_back(row);
        }
    }
}

void matrix_t::populate_best_col_neighbors(int anchor_col,int row){
    bool debug = false;
    if(debug) printf("best col neighbors at row %d\n",row);
    memset(best_col_neighbors,0,sizeof(bool)*m);
    //best_col_neighbors.clear();
    for(int_set_t::iterator it = positive_cols_indices2[row]->begin();
    it!=positive_cols_indices2[row]->end();++it){
        int col = *it;
        if(col!=anchor_col && intersecting_rows_indices_by_anchor2[anchor_col][col]->size()>=beta){
            if(debug)printf("Adding best col neighbor %d\n",col);
            //best_col_neighbors.insert(col);
            best_col_neighbors[col] = true;
        }
    }
}

void matrix_t::populate_best_neighbors(int col){
    bool debug = false;
    if(debug)printf("Best neighbors:");
    int count = 0;
    int j = col;
    int inc = 1;
    bool moveup = false;
    best_neighbors.clear();
    //return;
    while(true){
        if(j>=0 && j<m){
            if(best_col_neighbors[j]){
                for(int_vec_t::iterator it = best_row_neighbors.begin();
                it!=best_row_neighbors.end();++it){
                    int i = *it;
                    //if(positive_indices.find(make_pair(i,j))!=
                    //positive_indices.end()){
                    if(positive_indices_mat[idx(i,j)]){
                        best_neighbors.push_back(make_pair(i,j));
                        if(debug)printf(" %d,%d",i,j);
                        ++count;
                    }
                    if(count>limit) break;
                }
            }
        }
        if(count>limit || ((col+inc)>=m && (col-inc)<0 )) break;
        if(!moveup){
            j = col-inc;
            moveup = true;
        }else{
            j = col+inc;
            moveup = false;
            ++inc;
        }
    }
    if(debug)printf("\n");
    if(debug)printf("Total best neighbors: %d\n",(int)best_neighbors.size());
}

float matrix_t::compute_square(int row,int col){
    float res = 0;
    float lambda = 1;
    float total_weight = 0;
    bool debug = false;
    if(debug) printf("Compute_square\n");

    for(int_pair_vec_t::iterator it = best_neighbors.begin();
    it!=best_neighbors.end();++it){
        int i = it->first;
        int j = it->second;
        float row_result = 0.0;
        if(row_cache_used[idx(row,i)]){
            row_result = row_cache[idx(row,i)];
        }else{
            // compute row square
            int support = intersecting_cols_indices2[i]->size();
            for(int_vec_t::iterator it1 = intersecting_cols_indices2[i]->begin();
            it1!=intersecting_cols_indices2[i]->end();++it1){
                for(int_vec_t::iterator it2 = 
                intersecting_cols_indices2[i]->begin();
                it2!=intersecting_cols_indices2[i]->end();++it2){
                    int j1 = *it1;
                    int j2 = *it2;
                    float diff = 
                    (data[idx(row,j1)] - data[idx(i,j1)])
                    - (data[idx(row,j2)] - data[idx(i,j2)]);
                    row_result+= diff * diff;
                }
            }
            row_result/=2.0*support*(support-1);
            if(debug)printf("row result %.2f\n",row_result);
            row_cache[idx(row,i)]=row_result;
            row_cache_used[idx(row,i)]=true;
        }
        // compute col square
        float col_result = 0.0;
        if(col_cache_used[idx(col,j)]){
            col_result = col_cache[idx(col,j)];
        }else{
            int support = intersecting_rows_indices_by_anchor2[col][j]->size();
            for(int_vec_t::iterator it1 = 
            intersecting_rows_indices_by_anchor2[col][j]->begin();
            it1!=intersecting_rows_indices_by_anchor2[col][j]->end();++it1){
                for(int_vec_t::iterator it2 = 
                intersecting_rows_indices_by_anchor2[col][j]->begin();
                it2!=intersecting_rows_indices_by_anchor2[col][j]->end();++it2){
                    int i1 = *it1;
                    int i2 = *it2;
                    float diff = (data[idx(i1,col)] - data[idx(i1,j)])
                    - (data[idx(i2,col)] - data[idx(i2,j)]);
                    col_result+= diff * diff;
                }
            }

            col_result/=2.0*support*(support-1);
            if(debug)printf("col result %.2f\n",col_result);
            col_cache[idx(col,j)] = col_result;
            col_cache_used[idx(col,j)] = true;
        }
        float row_col_result = row_result<col_result?row_result:col_result;
        float weight_min = exp(-lambda * row_col_result);
        if(debug)printf("weight_min %.2f\n",weight_min);
        res+=weight_min* (data[idx(row,j)]+data[idx(i,col)]-data[idx(i,j)]);
        total_weight += weight_min;
    }
    return total_weight==0?0:res/total_weight;
}


float * matrix_t::estimate_gaussian(){
    bool debug = true;
    memset(row_cache,0.0,sizeof(float)*n*n);
    memset(row_cache_used,0,sizeof(bool)*n*n);
    memset(col_cache,0.0,sizeof(float)*m*m);
    memset(col_cache_used,0,sizeof(bool)*m*m);
    float * estimate = new float[n*m];
    memset(estimate,0,sizeof(float)*n*m);
    for(int row=0;row<n;++row){
        if(debug)printf("At row %d\n",row);
        populate_intersecting_cols(row);
        for(int col=0;col<m;++col){
            //if(debug)printf("At col %d\n",col);
            double start = clock();
            if(row==0){
                populate_intersecting_rows(col);
                //printf("A Elapsed: %.2f\n",(clock()-start));
            }
            //start = clock();
            populate_best_row_neighbors(row, col);
            //printf("B Elapsed: %.2f\n",(clock()-start));
            populate_best_col_neighbors(col, row);
            //printf("C Elapsed: %.2f\n",(clock()-start));
            populate_best_neighbors(col);
            //printf("D Elapsed: %.2f\n",(clock()-start));
            estimate[idx(row,col)] = compute_square(row,col);
            if(debug && (col % 1000 == 0)){
                printf(" col %d, best neighbors %d, estimate %.2f\n",col,(int)best_neighbors.size(),estimate[idx(row,col)]);
                printf("Elapsed: %.2f\n",(clock()-start));
            }
        }

    }
    return estimate;
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
    float * estimate = matrix->estimate_gaussian();
    FILE * pfile = fopen("estimate.txt","w");
    for(int i=0;i<n;++i){
        for(int j=0;j<m;++j){
            if(j) fprintf(pfile,"\t");
            fprintf(pfile,"%.2f",estimate[idx(i,j)]);
        }
        fprintf(pfile,"\n");
    }

    fclose(pfile);
    //int rank = 1;
    //impute(matrix,rank);
    //cleanup();
    delete matrix;
    //gpu_demo();
    return(EXIT_SUCCESS);
}
