movie = DirectShowSource("F:\Youtube\Events\animal\YT_animal_001.mpg")
return LanczosResize(ConvertToRGB32(movie), 160, 120)
