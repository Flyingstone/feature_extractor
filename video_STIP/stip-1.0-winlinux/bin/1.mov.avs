movie = DirectShowSource("3.mpg", fps=30)
return BilinearResize(ConvertToRGB24(movie), 160, 120)