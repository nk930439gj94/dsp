#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include "hmm.h"

using namespace std;

class Veterbi
{
public:
    Veterbi() { _hmm=0; _sequence=0; _delta=0; }
    ~Veterbi(){
        if( _sequence!=0 ) delete [] _sequence;
        if( _delta!=0 ){
            for(int i=0; i<_seqlength; ++i) delete [] _delta[i];
            delete [] _delta;
        }
    }
    void setmodel(HMM *h) { _hmm=h; }
    void initialize(unsigned len){
        if(_hmm==0){
            cerr<<"No model being set !\n";
            exit(1);
        }
        _seqlength = len;
        _sequence = new unsigned[_seqlength];

        _delta = new double*[_seqlength];
        for(int i=0; i<_seqlength; ++i) _delta[i] = new double[_hmm->state_num];
    }
    void read(const string &data){
        for(int i=0; i<_seqlength; ++i) _sequence[i] = int(data[i]) - int('A');
    }

    double get_highest_prob(){
        return _highest_prob;
    }

    void compute(){
        
        for(int i=0; i<_seqlength; ++i){
            for(int j=0; j<_hmm->state_num; ++j) _delta[i][j]=0;
        }

        for(int i=0; i<_hmm->state_num; ++i)
            _delta[0][i] = _hmm->initial[i] * _hmm->observation[_sequence[0]][i];
        double temp;
        for(int t=1; t<_seqlength; ++t){
            for(int i=0; i<_hmm->state_num; ++i){
                for(int j=0; j<_hmm->state_num; ++j){
                    temp = _delta[t-1][j] * _hmm->transition[j][i];
                    if( temp > _delta[t][i] ) _delta[t][i] = temp;
                }
                _delta[t][i] *= _hmm->observation[_sequence[t]][i];
            }
        }
        double max_prob=0;
        for(int i=0; i<_hmm->state_num; ++i){
            if( _delta[_seqlength-1][i] > max_prob ) max_prob = _delta[_seqlength-1][i];
        }
        _highest_prob = max_prob;
    }

private:
    HMM *_hmm;
    unsigned _seqlength;
    unsigned *_sequence;
    double **_delta;
    double _highest_prob;

    
};

int main(int argc,char **argv)
{
    if( argc!=4 ){
        cerr<<"Wrong command number !\n";
        exit(1);
    }
    vector<HMM*> models;
    ifstream ifs(argv[1]);
    if( !ifs.is_open() ){
        cerr<<"Cannot open file "<<argv[1]<<" !\n";
        exit(1);
    }
    string line;
    while( getline(ifs,line) ){
        models.push_back( new HMM );
        loadHMM( models[models.size()-1],line.c_str() );
    }
    ifs.close();

    ifs.open(argv[2]);
    if( !ifs.is_open() ){
        cerr<<"Cannot open file "<<argv[2]<<" !\n";
        exit(1);
    }
    Veterbi veterbi;
    vector<unsigned> results_model;
    vector<double> results_prob;
    bool first_time=true;
    while( getline(ifs,line) ){
        if(first_time){
            veterbi.setmodel(models[0]);
            veterbi.initialize(line.size());
        }
        unsigned highest_model=0;
        double highest_prob=0;
        veterbi.read(line);
        for(int i=0; i<models.size(); ++i){
            veterbi.setmodel( models[i] );
            veterbi.compute();
            if( veterbi.get_highest_prob() > highest_prob ){
                highest_model = i;
                highest_prob = veterbi.get_highest_prob();
            }
        }
        results_model.push_back(highest_model);
        results_prob.push_back(highest_prob);
        first_time = false;
    }
    ifs.close();


    ofstream ofs(argv[3]);
    if(!ofs.is_open()){
        cerr<<"Cannot open file "<<argv[3]<<" !\n";
        exit(1);
    }
    for(int i=0; i<results_model.size(); ++i)
        ofs<<models[ results_model[i] ]->model_name<<" "<<results_prob[i]<<endl;
    ofs.close();

    for(int i=0; i<models.size(); ++i) delete models[i];
}