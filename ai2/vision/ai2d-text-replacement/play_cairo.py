import cairo  # import the Python module
import cv2


img = cv2.imread('./ai2.png')

# setup a place to draw
# surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 200)
surface = cairo.ImageSurface.create_from_png("./ai2.png")
ctx = cairo.Context(surface)

# # paint background
# ctx.set_source_rgb(0.22, 0.08, 0.69)  # blue
# ctx.rectangle(0, 0, 400, 200)
# ctx.fill()

# draw text
ctx.select_font_face('Sans')
ctx.set_font_size(60)  # em-square height is 90 pixels
ctx.move_to(10, 90)  # move to point (x, y) = (10, 90)
ctx.set_source_rgb(1.00, 0.83, 0.00)  # yellow
ctx.show_text('Hello World')

# finish up
ctx.stroke()  # commit to surface
surface.write_to_png('hello_world.png')  # write to file


