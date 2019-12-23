#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    if( argc!=4 ){
        cerr<<"Wrong command number !\n";
        exit(1);
    }
    ifstream ifs_1(argv[1]);
    ifstream ifs_2(argv[2]);
    string line_1, line_2;
    unsigned count=0, same=0;
    while( getline(ifs_1,line_1) ){
        ++count;
        getline(ifs_2,line_2);
        line_1 = line_1.substr( 0,line_1.find(' ') );
        if(line_1==line_2) ++same;
    }
    ofstream ofs(argv[3]);
    ofs<<double(same)/count<<endl;
}