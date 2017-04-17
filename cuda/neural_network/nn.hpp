class nn_t{
public:
  nn_t(vector<int> inner_layer_dims,int label, float * first_layer);
  // NN stuff
  float get_error();
  void propagate_forward();
  ~nn_t();
  static void read_input(int n, int p,const char * labels, const char * features, int * & label_data, float * & feature_data){
    ifstream ifs_label,ifs_feature;
    ifs_label.open(labels);
    ifs_feature.open(features);
    if(!ifs_label.is_open() || !ifs_feature.is_open()){
      cerr<<"Cannot open label or feature file\n";
      exit(1);
    }
    label_data = new int[n];
    feature_data = new float[n*p];
    for(int i=0;i<n;++i){
      string label_line,feature_line;
      getline(ifs_label,label_line);
      getline(ifs_feature,feature_line);
      istringstream iss_label(label_line);
      istringstream iss_feature(feature_line);
      iss_label>>label_data[i];
      for(int j=0;j<p;++j){
        iss_feature>>feature_data[i*p+j];
      }
    }
    ifs_label.close();
    ifs_feature.close();

  }
private:
  // total inner layers
  int total_inner_layers;
  // the label is the training label
  int label;
  // the output is the final layer's neuron
  float output;
  // we treat the first inner layer as the actual input data
  vector<int> inner_layer_dims;
  // first dim is layer, second dim is neuron
  float ** inner_layer;
  // first dim is layer, second dim is output neuron, third dim is input neuron
  float *** weights;
  // BLAS stuff
  float inner_product(int n,float * v1,float * v2);
  float logistic_function(float x);

};
