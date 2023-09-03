// secref.cpp -- defining and using a reference
#include <iostream>
using namespace std;
int main()
{
    int rats = 101;
    int &rodents = rats; // rodents is a reference  声明的时候表示引用   其他情况表示取地址  同理&&表示右值引用
    cout << "rats =" << rats;
    cout << ",rodents =" << rodents << endl;
    cout << "rats address =" << &rats;
    cout << ",rodents address =" << &rodents << endl;
    int bunnies = 50;
    rodents = bunnies; // can we change the reference?cout <<"bunnies ="<< bunnies;cout <<",rats ="<< rats;cout <<",rodents ="<< rodents << endl;
    cout << "bunnies address =" << &bunnies;
    cout << ",rodents address =" << &rodents << endl;
    return 0;
}
