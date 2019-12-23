#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace std;

int main(int argc, char** argv)
{
    map<string, int> LineMapper;
    vector<string> FILE[14000];
    {
        ifstream Big5Zhu( argv[1] );
        int LineId = 0;
        string line, word;
        string Init[5];
        int NumInit, pos;
        bool isSameInit;
        while( getline(Big5Zhu, line) ){
            word = line.substr(0,2);
            isSameInit = false;
            NumInit = 0;
            for (int i = 2; i < line.size(); ++i) {
                if (line[i] == ' ' || line[i] == '/'){
                    if (NumInit == 0) { 
                        Init[0] = line.substr(i+1, 2);
                        if (LineMapper.count(Init[0]) == 0) {
                            LineMapper[Init[NumInit]] = LineId++;
                        }
                        ++NumInit; 
                    }
                    else {
                        for (int j = 0; j < NumInit; ++j) {
                            if (line.substr(i+1, 2) == Init[j]) {
                                isSameInit = true;
                                break;
                            }
                        }
                        if(!isSameInit) {
                            Init[NumInit] = line.substr(i+1, 2);
                            ++NumInit;
                        }
                        isSameInit = false;
                    }
                }
            }
            for (int i = 0; i < NumInit; ++i)
                FILE[LineMapper[Init[i]]].push_back(word);
            
            if(LineMapper.count(word) == 0) LineMapper[word] = (LineId++);
            FILE[LineMapper[word]].push_back(word);
        }
        Big5Zhu.close();
    }


    ofstream ZhuBig5( argv[2] );
    int th;
    for(auto it=LineMapper.begin(); it != LineMapper.end(); ++it) {
        th = it->second;
        ZhuBig5 << it->first;
        for(int i = 0; i < FILE[th].size(); ++i) {
            ZhuBig5 << " " << FILE[th][i];
        }
        ZhuBig5 << endl;
    }
    ZhuBig5.close();
}