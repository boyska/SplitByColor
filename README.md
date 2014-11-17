Split By Color
==============

A simple C program to cut ppm images along a line identified
by two green dots.

Purpose
--------

If you have a book scanner, just draw two green dots on it, shoot a photo, run
./split, enjoy having two .ppm


To compile
----------

Just `make`. It's plain C, with no external dependencies. have fun.

How to use
----------

./split -p -s -f <filename>.ppm

The code is fairly commented.
The code to read and write the PPM is taken from an
example in stackoverflow (http://stackoverflow.com/questions/2693631/read-ppm-file-and-store-it-in-an-array-coded-with-c).



TO DO:
------
- Integrate with ppmrose (https://aur.archlinux.org/packages/ppmrose) to perform the unwarping
- Test with real book scanner images
- A LOT of debug!
