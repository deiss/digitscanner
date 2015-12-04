#ifndef Window_hpp
#define Window_hpp

#include "DigitScanner.hpp"

class Window {

    public :
    
        Window(const int, const int);
        ~Window();

        void setDgs(DigitScanner<float>* dgs) { this->dgs = dgs; }
        void setSceneWidth(int scene_width)   { this->scene_width = scene_width; }
        void setSleepTime(int sleep_time)     { this->sleep_time = sleep_time; }

        void init();
        void launch() const;
 static void draw();
 static void keyboard(unsigned char, int, int);
 static void motion(int, int);
 static void passive(int, int);
 static void reshape(int, int);
    
    private :
 
 static int                  mouse_x;
 static int                  scene_width;
 static int                  sleep_time;
 static int                  window_height;
 static int                  window_width;
 static DigitScanner<float>* dgs;

};

#endif