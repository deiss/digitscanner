
//void call_from_main();
/* enable all numerical exceptions, for debugging */
// call_from_main();

#ifdef LINUX
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif
#endif
#include <fenv.h>
#define DEFINED_PPC      (defined(__ppc__) || defined(__ppc64__))
#define DEFINED_INTEL    (defined(__i386__) || defined(__x86_64__))
#ifndef LINUX
#if DEFINED_PPC
#define FE_EXCEPT_SHIFT 22
#define FM_ALL_EXCEPT    FE_ALL_EXCEPT >> FE_EXCEPT_SHIFT
static int
fegetexcept (void) {
    static fenv_t fenv;
    return (fegetenv (&fenv) ? -1 :((fenv & (FM_ALL_EXCEPT)) << FE_EXCEPT_SHIFT));
}
static int feenableexcept (unsigned int excepts) {
    static fenv_t fenv;
    unsigned int new_excepts = (excepts & FE_ALL_EXCEPT) >> FE_EXCEPT_SHIFT,
    old_excepts;  // all previous masks
    if ( fegetenv (&fenv) ) return -1;
    old_excepts = (fenv & FM_ALL_EXCEPT) << FE_EXCEPT_SHIFT;
    
    fenv = (fenv & ~new_excepts) | new_excepts;
    return (fesetenv (&fenv) ? -1 : old_excepts);
}
static int fedisableexcept (unsigned int excepts) {
    static fenv_t fenv;
    unsigned int still_on = ~( (excepts & FE_ALL_EXCEPT) >> FE_EXCEPT_SHIFT ),
    old_excepts;  // previous masks
    if ( fegetenv (&fenv) ) return -1;
    old_excepts = (fenv & FM_ALL_EXCEPT) << FE_EXCEPT_SHIFT;
    fenv &= still_on;
    return (fesetenv (&fenv) ? -1 : old_excepts);
}
#elif DEFINED_INTEL
static int fegetexcept (void) {
    static fenv_t fenv;
    return fegetenv (&fenv) ? -1 : (fenv.__control & FE_ALL_EXCEPT);
}
static int feenableexcept(unsigned int excepts) {
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
    old_excepts;  // previous masks
    if ( fegetenv (&fenv) ) return -1;
    old_excepts = fenv.__control & FE_ALL_EXCEPT;
    // unmask
    fenv.__control &= ~new_excepts;
    fenv.__mxcsr   &= ~(new_excepts << 7);
    return (fesetenv (&fenv) ? -1 : old_excepts);
}
static int fedisableexcept(unsigned int excepts) {
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
    old_excepts;  // all previous masks
    if ( fegetenv (&fenv) ) return -1;
    old_excepts = fenv.__control & FE_ALL_EXCEPT;
    // mask
    fenv.__control |= new_excepts;
    fenv.__mxcsr   |= new_excepts << 7;
    return (fesetenv (&fenv) ? -1 : old_excepts);
}
#endif  // PPC or INTEL enabling
#endif  // not LINUX
#if DEFINED_PPC
#define getfpscr(x)    asm volatile("mffs %0" : "=f" (x));
#define setfpscr(x)    asm volatile("mtfsf 255,%0" : : "f" (x));
typedef union {
    struct {
        unsigned long hi;
        unsigned long lo;
    } i;
    double d;
} hexdouble;
#endif  // DEFINED_PPC
#if DEFINED_INTEL
#define getx87cr(x)    asm("fnstcw %0" : "=m" (x));
#define setx87cr(x)    asm("fldcw %0"  : "=m" (x));
#define getx87sr(x)    asm("fnstsw %0" : "=m" (x));
#define getmxcsr(x)    asm("stmxcsr %0" : "=m" (x));
#define setmxcsr(x)    asm("ldmxcsr %0" : "=m" (x));
#endif  // DEFINED_INTEL
#include <signal.h>
#include <stdio.h>   // printf()
#include <stdlib.h>  // abort(), exit()
static char *fe_code_name[] = {
    const_cast<char *>("FPE_NOOP"),
    const_cast<char *>("FPE_FLTDIV"), const_cast<char *>("FPE_FLTINV"), const_cast<char *>("FPE_FLTOVF"), const_cast<char *>("FPE_FLTUND"),
    const_cast<char *>("FPE_FLTRES"), const_cast<char *>("FPE_FLTSUB"), const_cast<char *>("FPE_INTDIV"), const_cast<char *>("FPE_INTOVF"),
    const_cast<char *>("FPE_UNKNOWN")
};
static void fhdl ( int sig, siginfo_t *sip, ucontext_t *scp ) {
    int fe_code = sip->si_code;
    unsigned int excepts = fetestexcept(FE_ALL_EXCEPT);
    switch(fe_code) {
#ifdef FPE_NOOP  // occurs in OS X
        case FPE_NOOP:   fe_code = 0; break;
#endif
        case FPE_FLTDIV: fe_code = 1; break; // divideByZero
        case FPE_FLTINV: fe_code = 2; break; // invalid
        case FPE_FLTOVF: fe_code = 3; break; // overflow
        case FPE_FLTUND: fe_code = 4; break; // underflow
        case FPE_FLTRES: fe_code = 5; break; // inexact
        case FPE_FLTSUB: fe_code = 6; break; // invalid
        case FPE_INTDIV: fe_code = 7; break; // overflow
        case FPE_INTOVF: fe_code = 8; break; // underflow
        default: fe_code = 9;
    }
    if (sig==SIGFPE) {
#if DEFINED_INTEL
        unsigned short x87cr,x87sr;
        unsigned int mxcsr;
        getx87cr (x87cr);
        getx87sr (x87sr);
        getmxcsr (mxcsr);
        printf("X87CR:   0x%04X\n", x87cr);
        printf("X87SR:   0x%04X\n", x87sr);
        printf("MXCSR:   0x%08X\n", mxcsr);
#endif
#if DEFINED_PPC
        hexdouble t;
        getfpscr (t.d);
        printf("FPSCR:   0x%08X\n", t.i.lo);
#endif
        printf("signal:  SIGFPE with code %s\n", fe_code_name[fe_code]);
        printf("invalid flag:    0x%04X\n", excepts & FE_INVALID);
        printf("divByZero flag:  0x%04X\n", excepts & FE_DIVBYZERO);
    }
    else printf("Signal is not SIGFPE, it's %i.\n", sig);
    abort();
}
void call_from_main() {
    struct sigaction act;
    act.sa_sigaction = (void(*)(int, siginfo_t*, void*))fhdl;
    sigemptyset (&act.sa_mask);
    act.sa_flags = SA_SIGINFO;
    printf ("Old invalid exception:   0x%08X\n", feenableexcept (FE_INVALID));
    printf ("New fp exception:        0x%08X\n", fegetexcept ());
    if (sigaction(SIGFPE, &act, (struct sigaction *)0) != 0) {
        perror("Yikes");
        exit(-1);
    }
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    fedisableexcept(FE_ALL_EXCEPT);
}