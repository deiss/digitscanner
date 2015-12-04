#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#ifdef __linux__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#include "Window.hpp"

/* static variables */
DigitScanner<float>* Window::dgs;
int                  Window::mouse_x;
int                  Window::scene_width = 1;
int                  Window::sleep_time = 5;
int                  Window::window_width;
int                  Window::window_height;

/* Window constructor. */
Window::Window(const int w_width, const int w_height) {
    window_width  = w_width;
    window_height = w_height;
    mouse_x       = window_width/2;
}

/* Window destructor. */
Window::~Window() {
}

/* Initialization function. */
void Window::init() {
    int   argc    = 1;
    char *argv[1] = {(char *)"DigitScanner"};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(window_width, window_height);
}

/* Calls the Graph draw() function. */
void Window::draw() {
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3ub(0, 0, 100);
    dgs->draw();
    glutSwapBuffers();
    glutPostRedisplay();
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
}

/* Calls the Graph keyboard() function. */
void Window::keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case 'g' : dgs->guess(); break;
        case 'r' : dgs->reset(); break;
    }
}

/* Mouse function */
void Window::motion(int x, int y) {
    int i = static_cast<int>(y/10);
    int j = static_cast<int>(x/10);
    bool inside_window = true;
    if(i<0 || i>27 || j<0 || j>27) inside_window = false;
    if(inside_window) dgs->scan(i, j, 255);
}

/* Mouse function */
void Window::passive(int x, int y) {
    mouse_x = x;
}

/* Initializes new windows. */
void Window::launch() const {
    glutCreateWindow("DigitScanner");
    glViewport(0, 0, window_width, window_height);
    glClearColor(1, 1, 1, 1);
    glutReshapeFunc(reshape);
    glutDisplayFunc(draw);
    glutKeyboardFunc(keyboard);
    glutPassiveMotionFunc(passive);
    glutMotionFunc(motion);
    glutMainLoop();
}

/* Reshape function. */
void Window::reshape(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, scene_width, 0, scene_width*h/w);
}

