#!/usr/bin/python3
# -*- coding: utf-8 -*-

from enum import Enum


class Color(Enum):
    White = 1
    Black = 2
    Gray = 3
    Red = 4
    Blue = 5
    Green = 6
    Yellow = 7
    Purple = 8
    Brown = 9
    Orange = 10

    def get_color_literal(self):
        return color_dict[self]


class Shape(Enum):
    Circle = 1
    Semicircle = 2
    Quarter_Circle = 3
    Triangle = 4
    Square = 5
    Rectangle = 6
    Trapezoid = 7
    Pentagon = 8
    Hexagon = 9
    Heptagon = 10
    Octagon = 11
    Star = 12
    Cross = 13


Alphanum = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']

color_dict = {  # Sampled from https://www.auvsi-suas.org/
    Color.White:    '0xfff2f5',
    Color.Black:    '0x1d1d29',
    Color.Gray:     '0xcbcccb',
    Color.Red:      '0xe68a8c',
    Color.Blue:     '0x90a8d0',
    Color.Green:    '0x95b893',
    Color.Yellow:   '0xf7ef84',
    Color.Purple:   '0x9a81bb',
    Color.Brown:    '0xe3ba8f',
    Color.Orange:   '0xfbc077'
}