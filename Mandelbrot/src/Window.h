#pragma once

#include <memory>
#include <SDL.h>
#include <SDL_image.h>
#include <SDL_timer.h>
#include "Calc.h"

/*
* Deleters
*/
template <class T> struct Deleter;
template <> struct Deleter<SDL_Window> { void operator()(SDL_Window* p) const { SDL_DestroyWindow(p); } };
template <> struct Deleter<SDL_Renderer> { void operator() (SDL_Renderer* p) const { SDL_DestroyRenderer(p); } };
template <> struct Deleter<SDL_Surface> { void operator() (SDL_Surface* p) const { /*SDL_FreeSurface(p);*/ } };
template <> struct Deleter<SDL_Texture> { void operator() (SDL_Texture* p) const { SDL_DestroyTexture(p); } };


/*
* Unique Pointers
*/
template<class SDLType>
using Uptr = std::unique_ptr<SDLType, Deleter<SDLType>>;



/*
* Class Definitions
*/
class S
{
public:

    S();
    bool Init();

    void Run();
    void Update();
    void Render();

    ~S();

private:

    int width = 800;
    int height = 600;

    bool shutdown{ false };

    Uptr<SDL_Window> mWindow;
    Uptr<SDL_Renderer> mRenderer;
    Uptr<SDL_Surface> mSurface;
    Uptr<SDL_Texture> mTexture;

    MandelbrotCalc mMandelbrot;

};
