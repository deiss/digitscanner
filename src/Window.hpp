/*
DigitScanner - Copyright (C) 2016 - Olivier Deiss - olivier.deiss@gmail.com

DigitScanner is a C++ tool to create, train and test feedforward neural
networks (fnn) for handwritten number recognition. The project uses the
MNIST dataset to train and test the neural networks. It is also possible
to draw numbers in a window and ask the tool to guess the number you drew.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
This class deals with the rendering of the application. It receives the
events (mouse, keyboard) and prints the digit to screen.
*/

#ifndef Window_hpp
#define Window_hpp

#include "DigitScanner.hpp"

class Window {

    public :
    
        Window(const int, const int);
        ~Window() {}

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
 
 static int                  mouse_x;         /* x coordinate of the mouse */
 static int                  scene_width;     /* width of the scene */
 static int                  sleep_time;      /* sleeping time between two consecutive drawings (milliseconds) */
 static int                  window_height;   /* height of the window */
 static int                  window_width;    /* width of the window */
 static DigitScanner<float>* dgs;             /* associated DigitScanner object */

};

#endif
