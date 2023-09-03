#include <iostream>
using namespace std;

// 栈区数据注意事项：不要返回局部变量的地址
// 栈区的数据由编译器管理开辟和释放

int a = 10; 

int* func(int b) // 返回的是一个地址 int*；形参的数据也会放在栈区
{
    // 局部变量存放的数据10存放在栈区，栈区的数据在函数执行完后自动释放
    return &a;  // 返回局部变量的地址，即数据10的的地址
}

int *func_heap() // 返回的是一个地址 int*；形参的数据也会放在栈区
{
    // 利用new关键字，可以将数据开辟到堆区
    // 指针本质也是变量，这里的指针是局部变量，局部变量的数据放在栈上，即指针保存的数据是放在堆区
    int *p = new int(985); // new关键字会返回一个地址，因此用栈上的指针来接收堆上数据的地址。
    return p;
}
int main()
{

    cout << "hello world" << endl;
    // 接收func函数的返回值
    int *p = func(1); // 用指针接收栈区上的数据10的地址，由于栈区上数据10已经被释放，所以对地址解引用会获得乱码的值。

    cout << *p << endl; // 第一次可以 打印正确的数字，是因为编译器做了保留
    cout << *p << endl; // 第二次这个数据就不在保留了。

    // system("pause");

    // 在堆区开辟数据
    int *p2 = func_heap(); // 堆区的地址返回给 *p 了，栈区数据是否，堆区数据没释放

    cout << *p2 << endl;
    cout << *p2 << endl;
    cout << *p2 << endl;
    cout << *p2 << endl;

    return 0;
}