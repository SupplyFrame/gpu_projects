#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
#include<math.h>
#include<string.h>
#include<vector>

using namespace std;

#include "nn.hpp"


nn_t::nn_t(vector<int> inner_layer_dims,int label, float * first_layer){
  this->inner_layer_dims = inner_layer_dims;
  this->total_inner_layers = inner_layer_dims.size();
  //cerr<<"Total inner layers "<<total_inner_layers<<endl;
  this->label = label;
  this->inner_layer = new float*[total_inner_layers];
  this->weights = new float **[total_inner_layers];
  for(int layer=0;layer<total_inner_layers;++layer){
    int input_dim = inner_layer_dims[layer];
    int output_dim = layer==(total_inner_layers-1)?1:inner_layer_dims[layer+1];
    //cerr<<"At layer "<<layer<<" output dim "<<output_dim<<" input dim "<<input_dim<<endl;
    weights[layer]  = new float*[output_dim];
    for(int i=0;i<output_dim;++i){
      weights[layer][i]  = new float[input_dim];
      for(int j=0;j<input_dim;++j) {
        weights[layer][i][j] = 0.3;
        //cerr<<"At layer "<<layer<<" output neuron "<<i<<" input neuron "<<j<<endl;
      }
    }
    inner_layer[layer] = new float[input_dim];
  }
  // initialize first layer
  //cerr<<"Init first layer\n";
  for(int j=0;j<inner_layer_dims[0];++j){
    inner_layer[0][j] = first_layer[j];
  }
  cerr<<"Construction done\n";
}

float nn_t::inner_product(int n, float * v1,float * v2){
  float val = 0;
  for(int i=0;i<n;++i) val+=v1[i]*v2[i];
  //cerr<<"Inner product is "<<val<<endl;
  return val;
}

float nn_t::logistic_function(float x){
  float val = 1.0/1.0+exp(-x);
  //cerr<<"Logistic fxn is "<<val<<endl;
  return val;
}

void nn_t::propagate_forward(){
  cerr<<"Propagating forward\n";
  for(int layer=0;layer<total_inner_layers;++layer){
    int input_dim = inner_layer_dims[layer];
    int output_dim = layer==(total_inner_layers-1)?1:inner_layer_dims[layer+1];
    for(int output_neuron=0;output_neuron<output_dim;++output_neuron){
      float * layer_data = inner_layer[layer];
      float * layer_weight = weights[layer][output_neuron];
      float out = logistic_function(inner_product(input_dim,layer_data,layer_weight));
      //cerr<<"Output at layer "<<layer<<" output neuron "<<output_neuron<<" is "<<out<<endl;
      if(layer==total_inner_layers-1){
        //cerr<<"Setting final output\n";
        this->output = out;
      }else{
        //cerr<<"Setting inner layer output\n";
        inner_layer[layer+1][output_neuron] = out;
      }
      //cerr<<"Done.\n";
    }
  }
}

float nn_t::get_error(){
  float error = 1.0*(label-output)*(label-output);
  cerr<<"Label is "<<label<<" and output is "<<output<<endl;
  return error;
}

nn_t::~nn_t(){
  for(int layer=0;layer<total_inner_layers;++layer){
    //cerr<<"Destroying layer "<<layer<<endl;
    int output_dim = layer==(total_inner_layers-1)?1:inner_layer_dims[layer+1];
    //cerr<<"At layer "<<layer<<" output dim "<<output_dim<<endl;
    for(int i=0;i<output_dim;++i){
      delete[] weights[layer][i];
    }
    delete[] inner_layer[layer];
    //cerr<<"Destroyed layer "<<layer<<endl;
  }

  delete[] inner_layer;
  delete[] weights;
  cerr<<"Destroyed nn\n";
}


int main(int argc,char * argv[]){
  if (argc<6){
    cerr<<"Usage <examples> <label_file> <layer_1_file> <total_inner_layers k> <layer_1 dim> ... <layer_k dim> \n";
    return 1;
  }
  int arg=0;
  int total_examples = atoi(argv[++arg]);
  const char * labels_file=argv[++arg];
  const char * data_file=argv[++arg];
  int total_inner_layers=atoi(argv[++arg]);
  if(total_inner_layers<1){
    cerr<<"Total inner layers must be at least one.\n";
  }
  vector<int> inner_layer_dims; 
  for(int j=0;j<total_inner_layers;++j){
    inner_layer_dims.push_back(atoi(argv[++arg]));
    cerr<<"Inner layer "<<j<<" has dimension "<<inner_layer_dims[j]<<endl;
  }
  int total_features = inner_layer_dims[0];
  int * label_data = NULL;
  float * feature_data = NULL;
  nn_t::read_input(total_examples, total_features, labels_file, data_file, label_data, feature_data);
  nn_t** nn_arr = new nn_t*[total_examples];
  for(int i=0;i<total_examples;++i){
    cerr<<"Building neural network for example "<<i<<endl;
    nn_arr[i] = new nn_t(inner_layer_dims,label_data[i],
    feature_data+i*total_features);
    nn_arr[i]->propagate_forward();
    cerr<<"Error is "<<nn_arr[i]->get_error()<<endl;
  }
  // clean up
  for(int i=0;i<total_examples;++i){
    delete nn_arr[i];
  }

  delete[] nn_arr;
  if(label_data!=NULL) delete[] label_data;
  if(feature_data!=NULL) delete[] feature_data;
}
