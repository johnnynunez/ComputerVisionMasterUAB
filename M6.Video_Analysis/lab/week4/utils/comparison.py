#!/usr/bin/env python

import math

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import moviepy.editor
import numpy

intX = 32
intY = 436 - 64

objImages = [{
    'strFile': '/ghome/group03/mcv-m6-2023-team6/week4/Results/Task1_2/maskflownet/optical_flow_hsv.png',
    'strText': 'MaskFlowNet'
}, {
    'strFile': '/ghome/group03/mcv-m6-2023-team6/week4/Results/Task1_2/liteflownet/optical_flow_hsv.png',
    'strText': 'LiteFLowNet'
}, {
    'strFile': '/ghome/group03/mcv-m6-2023-team6/week4/Results/Task1_2/RAFT/optical_flow_hsv.png',
    'strText': 'RAFT'
}]

npyImages = []

for objImage in objImages:
    objOutput = PIL.Image.open(objImage['strFile'])

    imWidth, imHeight = objOutput.size

    for intU in [intShift - 10 for intShift in range(20)]:
        for intV in [intShift - 10 for intShift in range(20)]:
            if math.sqrt(math.pow(intU, 2.0) + math.pow(intV, 2.0)) <= 5.0:
                PIL.ImageDraw.Draw(objOutput).text((intX + intU, intY + intV), objImage['strText'], (255, 255, 255),
                                                   PIL.ImageFont.truetype('freefont/FreeSerifBold.ttf', 32))
            # end
        # end
    # end

    PIL.ImageDraw.Draw(objOutput).text((intX, intY), objImage['strText'], (0, 0, 0),
                                       PIL.ImageFont.truetype('freefont/FreeSerifBold.ttf', 32))

    npyImages.append(numpy.array(objOutput))
# end

moviepy.editor.ImageSequenceClip(sequence=npyImages, fps=0.5).write_gif(
    filename='/ghome/group03/mcv-m6-2023-team6/week4/Results/Task1_2/comparison.gif', fps=0.5)
