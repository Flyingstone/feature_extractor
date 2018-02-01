movie = DirectShowSource("3.flv", fps=25)
return LanczosResize(ConvertToRGB24(movie), 160, 120)
