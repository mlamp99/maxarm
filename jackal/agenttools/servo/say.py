#!/usr/bin/env python3
"""Sends one command to hold.py and prints its reply.   say.py down 0 -160 44"""
import socket
import sys

with socket.create_connection(("127.0.0.1", 8770), timeout=120) as s:
    s.sendall((" ".join(sys.argv[1:]) + "\n").encode())
    print(s.makefile().readline().strip())
