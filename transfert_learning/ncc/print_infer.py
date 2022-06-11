#!/usr/bin/env python3

import struct
f = open('test/mb.bin','rb')
# one image, 12 bytes
# 0000000 605c 3f7c c1b3 39fd faff 3c5f
# ie 3 fp32, ie softmax

data_raw = struct.unpack('f'*3,f.read(4*3))
f.close()
print(data_raw)


# (0.9858453273773193, 0.00048400237574242055, 0.013670681975781918)
