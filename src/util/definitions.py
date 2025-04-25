"""
Author: Lynn Ye
Created on: 2025/4/13
Brief: 
"""

IGNORE_LABEL_INDEX = -100

# Token types
NOTE_TYPE = 0
TS_TYPE = 1
VEL_TYPE = 2
DUM_TYPE = -1

# Since piano has pitch range 21-108, 0~20 are unused and can therefore be used as mask tokens
NOTE_MASK = 0
TIMESHIFT_MASK = -1 #
VELOCITY_MASK = -2
