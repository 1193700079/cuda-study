#include <iostream>
using namespace std;

// 引用做函数的返回值
// 1、不要返回局部变量的引用
int &test01()
{
    int a = 10; // 局部变量存放在四区中的栈区
    return a;
}

// 2、函数的调用可以作为左值
int &test02()
{
    static int a = 10; // 加上关键字static，变成静态变量，存放在全局区，全局区上的数据在程序结束后释放掉
    cout << "a = " << a << endl;
    return a;          // 函数的返回值是a的一个引用
}

int main()
{
    /*
    int& ref = test01();
    cout << "ref = " << ref << endl; //第一次结果正确，是因为编译器做了保留
    cout << "ref = " << ref << endl; //第一次结果正确，是因为栈区a的内存已经释放
    */

    int &ref = test02(); // 由于返回的是a的引用，所以要用引用来接收，这里用ref来接收，ref为原名a的别名
    cout << "ref = " << ref << endl;
    cout << "ref = " << ref << endl;
    cout << "ref = " << ref << endl;

    test02() = 1000;                 // 对a的引用进行操作，相当于原名a赋值赋值为1000
    cout << "ref = " << ref << endl; // 通过原名a的别名ref访问1000
    cout << "ref = " << ref << endl;

    system("pause");

    return 0;
}