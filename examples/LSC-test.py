from pvtrace import LSC
import time
import sys

lsc = LSC((5.0, 5.0, 1.0))
# Add solar cells to edge faces
lsc.add_solar_cell({'left', 'right', 'near', 'far'})
# NB solar cells are not rendered
lsc.show()
lsc.simulate(100)
lsc.report()

# Wait for Ctrl-C to terminate the script; keep the window open
print("Ctrl-C to close")
while True:
    try:
        time.sleep(.3)
    except KeyboardInterrupt:
        sys.exit()