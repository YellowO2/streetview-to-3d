"""What imagery exists in an area, before anything is reconstructed.

Asks Google and Apple what panoramas cover a corridor, on what dates, and
how they link to each other, then picks the dates that cover it best. No
GPU, no geometry -- the output is a graph of dots.
"""
