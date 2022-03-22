#include "Window.h"
#include <iostream>
#include <chrono>


S::S()
{
    mMandelbrot.AllocateBuffer(height, width);
}

S::~S()
{
    SDL_Quit();
}

void S::Run()
{
    while (!shutdown)
    {
        Update();
        Render();
    }
}

void S::Update()
{

    SDL_Event event;

    // Events management
    while (SDL_PollEvent(&event)) {

        const uint8_t* keyboard_state = SDL_GetKeyboardState(NULL);

        switch (event.type) {

        case SDL_QUIT:
            shutdown = true;
            break;


            // Keypresses
        case SDL_KEYDOWN:

            if (keyboard_state[SDL_SCANCODE_ESCAPE])
                shutdown = true;
            else if (keyboard_state[SDL_SCANCODE_W])
                mMandelbrot.ShiftYAxis(-1);
            else if (keyboard_state[SDL_SCANCODE_A])
                mMandelbrot.ShiftXAxis(-1);
            else if (keyboard_state[SDL_SCANCODE_S])
                mMandelbrot.ShiftYAxis(1);
            else if (keyboard_state[SDL_SCANCODE_D])
                mMandelbrot.ShiftXAxis(1);
            else if (keyboard_state[SDL_SCANCODE_UP])
                mMandelbrot.IncreaseIterations();
            else if (keyboard_state[SDL_SCANCODE_DOWN])
                mMandelbrot.DecreaseIterations();
            else if (keyboard_state[SDL_SCANCODE_LEFT])
                mMandelbrot.ChangeColor(1);
            else if (keyboard_state[SDL_SCANCODE_RIGHT])
                mMandelbrot.ChangeColor(-1);
            else if (keyboard_state[SDL_SCANCODE_SPACE])
                mMandelbrot.ChangeSet();


            break;

        case SDL_MOUSEWHEEL:

            int m_x, m_y;
            SDL_GetGlobalMouseState(&m_x, &m_y);
            float w = m_x > width ? 1.0f : (float)m_x / width;
            float h = m_y > height ? 1.0f : (float)m_y / height;

            // zoom in
            if (event.wheel.y > 0)
            {
                for (int i = 0; i < event.wheel.y; i++)
                {
                    mMandelbrot.ChangeXAxis(w);
                    mMandelbrot.ChangeYAxis(h);
                }
            }

            else if (event.wheel.y < 0)
            {
                for (int i = 0; i < abs(event.wheel.y); i++)
                {
                    mMandelbrot.ChangeXAxis(w, -1);
                    mMandelbrot.ChangeYAxis(h, -1);
                }
            }
            break;
        }
    }
}

void S::Render()
{

    if (mMandelbrot.Updated())
    {
        auto start = std::chrono::system_clock::now();
        mMandelbrot.Calculate();
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Rendering took " << static_cast<float>(duration) / 1000 << "s.\n";
 
        std::memcpy(mSurface.get()->pixels, mMandelbrot.GetBuffer(), mMandelbrot.BufferSize());
        mTexture = Uptr<SDL_Texture>(SDL_CreateTextureFromSurface(mRenderer.get(), mSurface.get()));
    }

    SDL_RenderClear(mRenderer.get());
    SDL_RenderCopy(mRenderer.get(), mTexture.get(), NULL, NULL);
    SDL_RenderPresent(mRenderer.get());
}

bool S::Init()
{
    if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
    {
        std::cerr << "Error initializing SDL\n";
    }

    mWindow = Uptr<SDL_Window>(SDL_CreateWindow("Mandelbrot", 100, 100, 800, 600, 0));
    mRenderer = Uptr<SDL_Renderer>(SDL_CreateRenderer(mWindow.get(), -1, SDL_RENDERER_ACCELERATED));
    mSurface = Uptr<SDL_Surface>(SDL_GetWindowSurface(mWindow.get()));

    return true;
}