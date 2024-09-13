#include <iostream>

using namespace std;

void printNums(int N);

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    printNums(N);

    return 0;
}

void printNums(int N){
    for (int i = 0; i <=N; i++) {
        cout << i << " ";
    }
    cout << endl;
    for (int i = N; i >=0; i--){
        cout << i << " ";
    }
}