# make gif
sudo ffmpeg -i $1_%d.png -vf palettegen /tmp/palette.png
sudo ffmpeg -i $1_%d.png -i /tmp/palette.png -lavfi paletteuse $1.gif
# make mp4
#sudo ffmpeg -framerate 14 -i $1_%d.png -f mp4 -pix_fmt yuv420p $1.mp4

