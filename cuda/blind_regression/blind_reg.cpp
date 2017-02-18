#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<map>
#include<set>
#include<ctime>
//#include<unordered_set>
#include<vector>

using namespace std;

typedef pair<int,int> int_pair_t;
typedef vector<int> int_vec_t;
typedef set<int> int_set_t;

typedef vector<int_vec_t> int_vec_vec_t ;
typedef vector<int_vec_vec_t> int_vec_vec_vec_t ;
typedef vector<int_set_t> int_set_vec_t ;
typedef vector<int_pair_t> int_pair_vec_t;
typedef set<int_pair_t> int_pair_set_t;
    
struct blind_reg_t{
    float beta;
    int limit;
    int n;
    int m;
    //float * data;
    float * data_rowmajor;
    float * data_colmajor;

    int_pair_set_t positive_indices;
    bool * positive_indices_mat;

    int_set_t ** positive_cols_indices_set;
    int_set_t ** positive_rows_indices_set;
    int_vec_t ** positive_cols_indices_vec;
    int_vec_t ** positive_rows_indices_vec;

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

    blind_reg_t(const char * inputfile,int rows,int cols);
    ~blind_reg_t();
    float * estimate_gaussian();
    void populate_intersecting_cols(int row);
    void populate_intersecting_rows(int col);
    void populate_best_row_neighbors(int row,int col);
    void populate_best_col_neighbors(int col,int row);
    void populate_best_neighbors(int col);
    float compute_square(int row,int col);
};


static void debug_matrix(const char *label,float * mat,int rows,int cols){
    printf("%s\n",label);
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            if(j) printf("\t");
            printf("%f",mat[j*rows+i]);
        }
        printf("\n");
    }
}

blind_reg_t::blind_reg_t(const char * inputfile,int rows,int cols){
    beta = 5.0;
    limit = 10000;
    n = rows;
    m = cols;
    //data = new float[rows*cols];
    //best_row_neighbors.reserve(n);
    //best_neighbors.reserve(limit);
    best_col_neighbors = new bool[m];
    data_rowmajor = new float[rows*cols];
    data_colmajor = new float[cols*rows];
    positive_cols_indices_set = new int_set_t*[n];
    positive_cols_indices_vec = new int_vec_t*[n];
    positive_rows_indices_set = new int_set_t*[m];
    positive_rows_indices_vec = new int_vec_t*[m];
    positive_indices_mat = new bool[n*m];
    memset(positive_indices_mat,0,sizeof(bool)*n*m);
    intersecting_cols_indices2 = new int_vec_t*[n];
    for(int i=0;i<n;++i){
        vector<int> v;
        set<int> s;
        intersecting_cols_indices2[i] = new vector<int>;
        //intersecting_cols_indices2[i]->reserve(m);
        positive_cols_indices_set[i] = new int_set_t;
        positive_cols_indices_vec[i] = new vector<int>;
        //positive_cols_indices_vec[i]->reserve(m);
    }
    intersecting_rows_indices_by_anchor2 = new int_vec_t**[m];
    for(int j=0;j<m;++j){
        set<int> s;
        positive_rows_indices_set[j] = new int_set_t;
        positive_rows_indices_vec[j] = new vector<int>;
        //positive_rows_indices_vec[j]->reserve(n);
        vector<int> v;
        intersecting_rows_indices_by_anchor2[j] = new vector<int>*[m];
        for(int k=0;k<m;++k){
            intersecting_rows_indices_by_anchor2[j][k] = new vector<int>;
            //intersecting_rows_indices_by_anchor2[j][k]->reserve(n);
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
            //data[idx(i,j)] = f;
            data_rowmajor[i*cols+j] = f; 
            data_colmajor[j*rows+i] = f; 
            if(f>0){
                positive_indices.insert(make_pair(i,j));
                positive_cols_indices_set[i]->insert(j);
                positive_cols_indices_vec[i]->push_back(j);
                positive_rows_indices_set[j]->insert(i);
                positive_rows_indices_vec[j]->push_back(i);
                positive_indices_mat[j*n+i] = true;
            }
        }
    }
    fclose(file_ptr);
}

blind_reg_t::~blind_reg_t(){
}

void blind_reg_t::populate_intersecting_cols(int row){
    bool debug = false;
    if(debug) printf("Intersecting cols at row %d\n",row);
    for(int i=0;i<n;++i){
        if(debug)printf("Populate intersecting on row %d anchor %d\n",i,row);
        intersecting_cols_indices2[i]->clear();
        for(int_vec_t::iterator it = positive_cols_indices_vec[i]->begin(),end=positive_cols_indices_vec[i]->end(); it!=end;++it){
            int col = *it;
            if(positive_cols_indices_set[row]->find(col)!=positive_cols_indices_set[row]->end()){
                if(debug)printf("Adding at col %d\n",col);
                intersecting_cols_indices2[i]->push_back(col);
            }
        }
    }
}

void blind_reg_t::populate_intersecting_rows(int col){
    bool debug = false;
    if(debug) printf("Intersecting rows at anchor col %d\n",col);
    for(int j=0;j<m;++j){
        if(debug)printf("Populate intersecting on col %d anchor %d\n",j,col);
        intersecting_rows_indices_by_anchor2[col][j]->clear();
        for(int_vec_t::iterator it = positive_rows_indices_vec[j]->begin(),end=positive_rows_indices_vec[j]->end(); it!=end;++it){
            int row = *it;
            if(positive_rows_indices_set[col]->find(row)!=positive_rows_indices_set[col]->end()){
                intersecting_rows_indices_by_anchor2[col][j]->push_back(row);
            }
            if(debug)printf("Col %d is size %d\n",j,(int)intersecting_rows_indices_by_anchor2[col][j]->size());
        }
    }
}

void blind_reg_t::populate_best_row_neighbors(int anchor_row,int col){
    bool debug = false;
    if(debug) printf("At beta %.2f best row neighbors at col %d\n",beta,col);
    best_row_neighbors.clear();
    for(int_vec_t::iterator it = positive_rows_indices_vec[col]->begin(),
    end=positive_rows_indices_vec[col]->end(); it!=end;++it){
        int row = *it;
        if(row!=anchor_row && intersecting_cols_indices2[row]->size()>=beta){
            if(debug)printf("Adding best row neighbor %d\n",row);
            best_row_neighbors.push_back(row);
        }
    }
}

void blind_reg_t::populate_best_col_neighbors(int anchor_col,int row){
    bool debug = false;
    if(debug) printf("best col neighbors at row %d\n",row);
    memset(best_col_neighbors,0,sizeof(bool)*m);
    for(int_vec_t::iterator it = positive_cols_indices_vec[row]->begin(),end=positive_cols_indices_vec[row]->end(); it!=end;++it){
        int col = *it;
        if(col!=anchor_col && intersecting_rows_indices_by_anchor2[anchor_col][col]->size()>=beta){
            if(debug)printf("Adding best col neighbor %d\n",col);
            //best_col_neighbors.insert(col);
            best_col_neighbors[col] = true;
        }
    }
}

void blind_reg_t::populate_best_neighbors(int col){
    bool debug = false;
    if(debug)printf("Best neighbors:");
    int count = 0;
    int j = col;
    int inc = 1;
    bool moveup = false;
    best_neighbors.clear();
    while(true){
        if(j>=0 && j<m){
            if(best_col_neighbors[j]){
                for(int_vec_t::iterator it = best_row_neighbors.begin(),end=best_row_neighbors.end(); it!=end;++it){
                    int i = *it;
                    if(positive_indices_mat[j*n+i]){
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

float blind_reg_t::compute_square(int row,int col){
    float res = 0;
    float lambda = 1;
    float total_weight = 0;
    bool debug = false;
    if(debug) printf("Compute_square\n");

    for(int_pair_vec_t::iterator it = best_neighbors.begin(),end=best_neighbors.end(); it!=end;++it){
        int i = it->first;
        int j = it->second;
        float row_result = 0.0;
        if(row_cache_used[row*n+i]){
            row_result = row_cache[row*n+i];
        }else{
            // compute row square
            int support = intersecting_cols_indices2[i]->size();
            for(int_vec_t::iterator it1 = intersecting_cols_indices2[i]->begin(),end=intersecting_cols_indices2[i]->end(); it1!=end;++it1){
                for(int_vec_t::iterator it2 = 
                intersecting_cols_indices2[i]->begin(),end=intersecting_cols_indices2[i]->end(); it2!=end;++it2){
                    int j1 = *it1;
                    int j2 = *it2;
                    float diff = 
                    (data_rowmajor[row*m+j1] - data_rowmajor[i*m+j1])
                    - (data_rowmajor[row*m+j2] - data_rowmajor[i*m+j2]);
                    //(data[idx(row,j1)] - data[idx(i,j1)])
                    //- (data[idx(row,j2)] - data[idx(i,j2)]);
                    row_result+= diff * diff;
                }
            }
            row_result/=2.0*support*(support-1);
            if(debug)printf("row result %.2f\n",row_result);
            row_cache[row*n+i]=row_result;
            row_cache_used[row*n+i]=true;
        }
        // compute col square
        float col_result = 0.0;
        if(col_cache_used[col*m+j]){
            col_result = col_cache[col*m+j];
        }else{
            int support = intersecting_rows_indices_by_anchor2[col][j]->size();
            for(int_vec_t::iterator it1 = intersecting_rows_indices_by_anchor2[col][j]->begin(),end=intersecting_rows_indices_by_anchor2[col][j]->end(); it1!=end;++it1){
                for(int_vec_t::iterator it2 = intersecting_rows_indices_by_anchor2[col][j]->begin(),end=intersecting_rows_indices_by_anchor2[col][j]->end(); it2!=end;++it2){
                    int i1 = *it1;
                    int i2 = *it2;
                    float diff = (data_colmajor[col*n+i1] - data_colmajor[j*n+i1])
                    - (data_colmajor[col*n+i2] - data_colmajor[j*n+i2]);
                    //float diff = (data[idx(i1,col)] - data[idx(i1,j)])
                    //- (data[idx(i2,col)] - data[idx(i2,j)]);
                    col_result+= diff * diff;
                }
            }

            col_result/=2.0*support*(support-1);
            if(debug)printf("col result %.2f\n",col_result);
            col_cache[col*m+j] = col_result;
            col_cache_used[col*m+j] = true;
        }
        float row_col_result = row_result<col_result?row_result:col_result;
        float weight_min = exp(-lambda * row_col_result);
        if(debug)printf("weight_min %.2f\n",weight_min);
        res+=weight_min* (data_rowmajor[row*m+j]+data_colmajor[col*n+i]-data_rowmajor[i*m+j]);
        total_weight += weight_min;
    }
    return total_weight==0?0:res/total_weight;
}


float * blind_reg_t::estimate_gaussian(){
    bool debug = true;
    memset(row_cache,0.0,sizeof(float)*n*n);
    memset(row_cache_used,0,sizeof(bool)*n*n);
    memset(col_cache,0.0,sizeof(float)*m*m);
    memset(col_cache_used,0,sizeof(bool)*m*m);
    float * estimate = new float[n*m];
    memset(estimate,0,sizeof(float)*n*m);
    for(int col=0;col<m;++col){
        populate_intersecting_rows(col);
    }
    for(int row=0;row<n;++row){
        if(debug)printf("At row %d\n",row);
        populate_intersecting_cols(row);
        for(int col=0;col<m;++col){
            //if(debug)printf("At col %d\n",col);
            double start = clock();
            //if(row==0){
                //printf("A Elapsed: %.2f\n",(clock()-start));
            //}
            //start = clock();
            populate_best_row_neighbors(row, col);
            //printf("B Elapsed: %.2f\n",(clock()-start));
            populate_best_col_neighbors(col, row);
            //printf("C Elapsed: %.2f\n",(clock()-start));
            populate_best_neighbors(col);
            //printf("D Elapsed: %.2f\n",(clock()-start));
            estimate[row*m+col] = compute_square(row,col);
            if(debug && (col % 1000 == 0)){
                printf(" col %d, best neighbors %d, estimate %.2f\n",col,(int)best_neighbors.size(),estimate[row*m+col]);
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
    blind_reg_t * blind_reg = new blind_reg_t(inputfile,n,m);
    float * estimate = blind_reg->estimate_gaussian();
    FILE * pfile = fopen("estimate.txt","w");
    for(int i=0;i<n;++i){
        for(int j=0;j<m;++j){
            if(j) fprintf(pfile,"\t");
            fprintf(pfile,"%.2f",estimate[i*m+j]);
        }
        fprintf(pfile,"\n");
    }

    fclose(pfile);
    delete blind_reg;
    return(EXIT_SUCCESS);
}
