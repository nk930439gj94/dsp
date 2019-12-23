#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include "Ngram.h"

using namespace std;

int main(int argc, char** argv)
{
    ifstream text, ZhuYin_Big5_map;
    for (int i = 1; i < argc; ++i){
        if (string(argv[i]) == "-text") text.open(argv[++i]);
        else if (string(argv[i]) == "-map") ZhuYin_Big5_map.open(argv[++i]);
    }
    Vocab voc;
    Ngram lm (voc, 2);
    {
        File lmFile( "./bigram.lm", "r" );
        lm.read(lmFile);
        lmFile.close();
    }

    vector<string> Pos2Words[14000];
    map<string, int> ZhuYin2Pos;
    {
        int LineNo = 0;
        string line, key, word;
        while(getline(ZhuYin_Big5_map, line)){
            key = line.substr(0, 2);
            if(ZhuYin2Pos.count(key) == 0) ZhuYin2Pos[key] = (LineNo++);
            istringstream words( line.substr(3) );
            while(getline(words, word, ' ')) Pos2Words[ZhuYin2Pos[key]].push_back(word);
        }
        ZhuYin_Big5_map.close();
    }

    string line;
    vector<string> words, trace;
    while(getline(text, line)) {
        words.clear();
        for (int i = 0; i < line.size(); ++i){
            if (line[i] != ' ') words.push_back(line.substr(i++, 2));
        }

        vector<double> prob[words.size()];
        vector<int> traceback[words.size()];
        for(int i = 0; i < Pos2Words[ZhuYin2Pos[words[0]]].size(); ++i){
            VocabIndex wordId = voc.getIndex(Pos2Words[ZhuYin2Pos[words[0]]][i].c_str());
            if (wordId == Vocab_None) wordId = voc.getIndex(Vocab_Unknown);
            VocabIndex context[] = { Vocab_None, Vocab_None };
            prob[0].push_back(lm.wordProb(wordId, context));
            traceback[0].push_back(0);
        }
        
        for (int i = 1; i < words.size(); ++i) {
            vector<string>& candidates_curr = Pos2Words[ZhuYin2Pos[words[i]]], &candidates_prev = Pos2Words[ZhuYin2Pos[words[i-1]]];
            for (int j = 0; j < candidates_curr.size(); ++j){
                VocabIndex wordId_curr = voc.getIndex(candidates_curr[j].c_str());
                if(wordId_curr == Vocab_None) wordId_curr = voc.getIndex(Vocab_Unknown);
                double delta = -1000000;
                int tracebackId = 0;
                for (int k = 0; k < candidates_prev.size(); ++k){
                    VocabIndex wordId_prev = voc.getIndex(candidates_prev[k].c_str());
                    if(wordId_prev == Vocab_None) wordId_prev = voc.getIndex(Vocab_Unknown);
                    VocabIndex context[] = { wordId_prev, Vocab_None };
                    double newdelta = prob[i-1][k] + lm.wordProb(wordId_curr, context);
                    if (newdelta > delta) {
                        delta = newdelta;
                        tracebackId = k;
                    } 
                }
                prob[i].push_back(delta);
                traceback[i].push_back(tracebackId);
            }
        }
        vector<string>& candidates_last = Pos2Words[ZhuYin2Pos[words.back()]];
        double max = prob[words.size()-1][0];
        int maxId = 0;
        for (int i = 1; i < candidates_last.size(); ++i) {
            if (prob[words.size()-1][i] > max) {
                max = prob[words.size()-1][i];
                maxId = i;
            }
        }
        trace.clear();
        trace.push_back(candidates_last[maxId]);
        for (int i = words.size()-2; i >= 0; --i) {
            trace.push_back(Pos2Words[ZhuYin2Pos[words[i]]][traceback[i+1][maxId]]);
            maxId = traceback[i+1][maxId];
        }
        
        cout << "<s> ";
        for (int i = trace.size()-1; i >= 0; --i) cout << trace[i] << " ";
        cout << "</s>" << endl;
    }
    text.close();
}