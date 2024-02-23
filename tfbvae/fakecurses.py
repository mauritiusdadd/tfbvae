#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:31:25 2021

@author: daddona
"""
import os
import sys
import threading

try:
    import curses
    HAS_CURSES = True
except ImportError:
    HAS_CURSES = False

try:
    from IPython.display import clear_output
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

def isnotebook():
    """
    Check if the code is run in a standard interpreter or in a
    IPython shell / Jupyter notebook

    Returns
    -------
    shell_type:
        - None if code is running in a standard interpreter
        - True if code is running in a jupyter noteboook
        - False if code is running in a ipython shell
        - False otherwise

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            shell_type = True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            shell_type = False
        else:
            # Other type (?)
            shell_type = False
    except NameError:
        # Probably standard Python interpreter
        shell_type = None

    return shell_type


##############################################
# Fake curse interface                       #
##############################################

# Global buffer for FakeCurses
_lines_buffer = {}

# Threading semaphore for FakeCurses
_lines_buffer_lock = threading.Condition()


def ncurseinit():
    if HAS_CURSES and isnotebook() is None:
        stdscr = curses.initscr()
        curses.noecho()
        stdscr.keypad(True)
        curses.curs_set(0)
        try:
            curses.cbreak()
        except Exception:
            print("cbreak mode unavailable")
        return stdscr
    else:
        return FakeCurses()


def ncursereset(stdscr):
    if HAS_CURSES and stdscr is not None:
        if isinstance(stdscr, FakeCurses):
            return
        try:
            curses.nocbreak()
        except Exception:
            print("buffer mode unavailable")
        curses.curs_set(1)
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


def ncursedoupdate(stdscr):
    if HAS_CURSES and stdscr is not None:
        if isinstance(stdscr, FakeCurses):
            stdscr.doupdate()
        else:
            curses.doupdate()


class FakeCurses():

    def __init__(self):
        self.lines = []

    def keypad(self, *args, **kargs):
        # Not implemented
        pass

    def addstr(self, y, x, text, fmt=None):
        self.lines.append((y, x, text))

    def noutrefresh(self):
        global _lines_buffer
        global _lines_buffer_lock
        _lines_buffer_lock.acquire()
        for y, x, text in self.lines:
            if y not in _lines_buffer:
                _lines_buffer[y] = text
            else:
                old_line = _lines_buffer[y]
                if x <= len(old_line):
                    new_line = old_line[:x]
                    new_line += text
                else:
                    new_line = old_line
                    new_line += " "*(x - len(old_line))
                    new_line += text
        _lines_buffer_lock.release()
        self.lines = []

    def doupdate(self):
        global _lines_buffer
        global _lines_buffer_lock
        buffer = ""
        _lines_buffer_lock.acquire()
        for i in range(max(_lines_buffer.keys())+1):
            try:
                new_line = _lines_buffer[i]
            except KeyError:
                pass
            else:
                buffer += new_line
            buffer += "\r\n"

        if isnotebook() is not None and HAS_IPYTHON:
            clear_output()

        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

        sys.stdout.write(buffer)
        sys.stdout.flush()
        _lines_buffer_lock.release()

    def clear(self):
        global _lines_buffer
        _lines_buffer = {}

    def refresh(self):
        self.noutrefresh()
        self.doupdate()
