#include "Window.h"

#undef main
int main(int argc, char* argv[])
{
    S s;
    if (s.Init())
        s.Run();
    return 0;
}
