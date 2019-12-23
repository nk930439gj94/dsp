#include <iostream>
#include <fstream>
#include <string>
#include "hmm.h"

using namespace std;


class train
{
public:
    train(HMM *h) { _hmm=h; _sequence=0; _alpha=0; _beta=0; _gamma=0; _epsilon=0; _b=0; }
    ~train(){
        if(_sequence!=0) delete _sequence;
        if(_alpha!=0){
            for(int i=0; i<_seqlength; ++i) delete [] _alpha[i];
            delete [] _alpha;
        }
        if(_beta!=0){
            for(int i=0; i<_seqlength; ++i) delete [] _beta[i];
            delete [] _beta;
        }
        if(_gamma!=0) delete [] _gamma;
        if(_epsilon!=0){
            for(int i=0; i<_hmm->state_num; ++i) delete [] _epsilon[i];
            delete [] _epsilon;
        }
        if(_b!=0){
            for(int i=0; i<_hmm->observ_num; ++i) delete [] _b[i];
            delete [] _b;
        }
        if(_initial!=0) delete [] _initial;
        if(_final!=0) delete [] _final;
    }

    void read(const string&);
    void rewrite(const string &data){
        for(int i=0; i<_seqlength; ++i) _sequence[i] = int(data[i])-int('A');
    }

    void compute_alpha();
    void compute_beta();
    void compute_gamma();   // call it after alpha beta are computed
    void compute_epsilon(); // call it after alpha beta are computed
    void compute_b();       // call it after gamma is computed
    void compute_initial();
    void compute_final();

    void compute_all(){
        compute_alpha();
        compute_beta();
        compute_gamma();
        compute_epsilon();
        compute_b();
        compute_initial();
        compute_final();
    }

    double* get_gamma() { return _gamma; }
    double** get_epsilon() { return _epsilon; }
    double** get_b() { return _b; }
    double* get_initial() { return _initial; }
    double* get_final() { return _final; }


    void test(){
        cout<<"initial:"<<endl;
        for(int i=0; i<_hmm->state_num; ++i){
            cout<<_hmm->initial[i]<<' ';
        }
        cout<<endl<<"transition:"<<endl;
        for(int i=0; i<_hmm->state_num; ++i){
            for(int j=0; j<_hmm->state_num; ++j)
                cout<<_hmm->transition[i][j]<<' ';
            cout<<endl;
        }
        cout<<"observation:"<<endl;
        for(int i=0; i<_hmm->observ_num; ++i){
            for(int j=0; j<_hmm->state_num; ++j)
                cout<<_hmm->observation[i][j]<<' ';
            cout<<endl;
        }
    }


private:
    HMM *_hmm;

    unsigned _seqlength;
    unsigned *_sequence;
    double **_alpha;    // _alpha[t][i]: in state i at time t
    double **_beta;
    double *_gamma;
    double **_epsilon;
    double **_b;        // _b[k][i]: gamma[i] specifying oberservation=k
    double *_initial;   // gamma t=1
    double *_final;     // gamma t=T
};

void train::read(const string &data){
    _seqlength = data.size();
    _sequence = new unsigned[_seqlength];
    for(int i=0; i<_seqlength; ++i) _sequence[i] = int(data[i])-int('A');
    _alpha = new double*[_seqlength];
    _beta = new double*[_seqlength];
    for(int i=0; i<_seqlength; ++i){
        _alpha[i] = new double[_hmm->state_num];
        _beta[i] = new double[_hmm->state_num];
    }
    _gamma = new double[_hmm->state_num];
    _epsilon = new double*[_hmm->state_num];
    for(int i=0; i<_hmm->state_num; ++i) _epsilon[i] = new double[_hmm->state_num];
    _b = new double*[_hmm->observ_num];
    for(int i=0; i<_hmm->observ_num; ++i) _b[i] = new double[_hmm->state_num];
    _initial = new double[_hmm->state_num];
    _final = new double[_hmm->state_num];
}

void train::compute_alpha()
{
    for(int i=0; i<_hmm->state_num; ++i)
        _alpha[0][i] = _hmm->initial[i] * _hmm->observation[_sequence[0]][i];
    for(int t=1; t<_seqlength; ++t){
        for(int i=0; i<_hmm->state_num; ++i){
            _alpha[t][i] = 0;
            for(int j=0; j<_hmm->state_num; ++j)
                _alpha[t][i] += _alpha[t-1][j] * _hmm->transition[j][i];
            _alpha[t][i] *= _hmm->observation[_sequence[t]][i];
        }
    }
}

void train::compute_beta()
{
    for(int i=0; i<_hmm->state_num; ++i)
        _beta[_seqlength-1][i] = 1;
    for(int t=_seqlength-2; t>=0; --t){
        for(int i=0; i<_hmm->state_num; ++i){
            _beta[t][i] = 0;
            for(int j=0; j<_hmm->state_num; ++j)
                _beta[t][i] += _hmm->transition[i][j] * _hmm->observation[_sequence[t+1]][j] * _beta[t+1][j];
        }
    }
}

void train::compute_gamma()
{
    double dominator = 0;
    for(int i=0; i<_hmm->state_num; ++i) dominator += _alpha[_seqlength-1][i];

    for(int i=0; i<_hmm->state_num; ++i){
        _gamma[i]=0;
        for(int t=0; t<_seqlength-1; ++t) _gamma[i] += _alpha[t][i] * _beta[t][i];
        _gamma[i] /= dominator;
    }
}

void train::compute_epsilon()
{
    double dominator = 0;
    for(int i=0; i<_hmm->state_num; ++i) dominator += _alpha[_seqlength-1][i];

    for(int i=0; i<_hmm->state_num; ++i){
        for(int j=0; j<_hmm->state_num; ++j){
            _epsilon[i][j] = 0;
            for(int t=0; t<_seqlength-1; ++t )
                _epsilon[i][j] += _alpha[t][i] * _hmm->transition[i][j] * _hmm->observation[_sequence[t+1]][j] * _beta[t+1][j];
            _epsilon[i][j] /= dominator;
        }
    }
}

void train::compute_b()
{
    double dominator = 0;
    for(int i=0; i<_hmm->state_num; ++i) dominator += _alpha[_seqlength-1][i];

    for(int i=0; i<_hmm->observ_num; ++i){
        for(int j=0; j<_hmm->state_num; ++j) _b[i][j] = 0;
    }
    for(int i=0; i<_hmm->state_num; ++i){
        for(int t=0; t<_seqlength; ++t) _b[_sequence[t]][i] += _alpha[t][i] * _beta[t][i];
    }
    for(int i=0; i<_hmm->observ_num; ++i){
        for(int j=0; j<_hmm->state_num; ++j) _b[i][j] /= dominator;
    }
}

void train::compute_initial()
{
    double dominator = 0;
    for(int i=0; i<_hmm->state_num; ++i) dominator += _alpha[_seqlength-1][i];
    for(int i=0; i<_hmm->state_num; ++i) _initial[i] = _alpha[0][i] * _beta[0][i] / dominator;
}

void train::compute_final()
{
    double dominator = 0;
    for(int i=0; i<_hmm->state_num; ++i) dominator += _alpha[_seqlength-1][i];
    for(int i=0; i<_hmm->state_num; ++i) _final[i] = _alpha[_seqlength-1][i] * _beta[_seqlength-1][i] / dominator;
}



int main(int argc,char **argv)
{
    const unsigned Iteration = stoul(argv[1]);
    HMM *model=new HMM;
    loadHMM(model,argv[2]);
    ifstream ifs(argv[3]);

    for(unsigned iter=0; iter<Iteration; ++iter){
        string line;
        unsigned data_size=0;
        train train(model);
        double *gamma_total=0;
        double **epsilon_total=0;
        double **b_total=0;
        double *initial_total=0;
        double *final_total=0;
        while(getline(ifs,line)){
            if(data_size == 0){
                train.read(line);
                gamma_total = new double[model->state_num]();
                epsilon_total = new double*[model->state_num];
                for(int i=0; i<model->state_num; ++i) epsilon_total[i] = new double[model->state_num]();
                b_total = new double*[model->observ_num];
                for(int i=0; i<model->observ_num; ++i) b_total[i] = new double[model->state_num]();
                initial_total = new double[model->state_num]();
                final_total = new double[model->state_num]();
            }
            else
                train.rewrite(line);
            
            train.compute_all();

            for(int i=0; i<model->state_num; ++i)
                gamma_total[i] += train.get_gamma()[i];
            for(int i=0; i<model->state_num; ++i){
                for(int j=0; j<model->state_num; ++j)
                    epsilon_total[i][j] += train.get_epsilon()[i][j];
            }
            for(int i=0; i<model->observ_num; ++i){
                for(int j=0; j<model->state_num; ++j)
                    b_total[i][j] += train.get_b()[i][j];
            }
            for(int i=0; i<model->state_num; ++i)
                initial_total[i] += train.get_initial()[i];
            for(int i=0; i<model->state_num; ++i)
                final_total[i] += train.get_final()[i];
            
            ++data_size;
        }

        for(int i=0; i<model->state_num; ++i)
            model->initial[i] = initial_total[i] / data_size;
        for(int i=0; i<model->state_num; ++i){
            for(int j=0; j<model->state_num; ++j)
                model->transition[i][j] = epsilon_total[i][j] / gamma_total[i];
        }
        for(int i=0; i<model->observ_num; ++i){
            for(int j=0; j<model->state_num; ++j)
                model->observation[i][j] = b_total[i][j] / ( gamma_total[j]+final_total[j] );
        }

        if( gamma_total!=0 ) delete [] gamma_total;
        if( epsilon_total!=0 ){
            for(int i=0; i<model->state_num; ++i)
                delete [] epsilon_total[i];
            delete [] epsilon_total;
        }
        if( b_total!=0 ){
            for(int i=0; i<model->observ_num; ++i)
                delete [] b_total[i];
            delete [] b_total;
        }
        if( initial_total!=0 ) delete [] initial_total;
        if( final_total!=0 ) delete [] final_total;

        ifs.clear();
        ifs.seekg(0,ifs.beg);
    }
    ifs.close();

    FILE *fp = open_or_die( argv[4], "w");
    dumpHMM(fp,model);
    delete model;
}