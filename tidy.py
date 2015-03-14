from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt, floor, ceil
from skimage.color import rgb2gray
from skimage import novice
import numpy

fn = 'NASA-HS201427a-HubbleUltraDeepField2014-20140603.jpg'

image = novice.open(fn).rgb
image_gray = rgb2gray(image)

###--- default is Difference of Hessians
###--- the Laplacian of Gaussians (log) is more accurate, but way way slower.

blobs = blob_doh(image_gray, max_sigma=25, threshold=.01, num_sigma=10)
radii = blobs[:,2]
r_bins = numpy.unique( radii )[::-1] 

max_count_full = 100
count_x = 0
last_x = 0
x_lim = []
image_arr = []

for radius in r_bins:
    count = 0
    full_count = 0
    count_x += 1
    max_count = floor( sqrt(numpy.sum(radii==radius) ) )

    ###--- if you have lots of singletons you could uncomment this.
    #if numpy.sum(radii==radius) <= 2:
    #    continue
    
    for blob in blobs:
        y, x, r = blob
        if r == radius:
            ##-- scale a nice window
            r += 1
            r *= 2
            sub_im = numpy.rot90(image[y-r:y+r, x-r:x+r])
            if numpy.sum(sub_im) != 0:
                ##-- check that this isn't spurious
                image_arr.append( [sub_im, (last_x, last_x+2*r, count*2*r, (count+1)*2*r) ] )
                count += 1
                full_count += 1
                x_lim.append( (count+1)*2*r )
            if full_count >= max_count*max_count:
                break
            
            if count >= max_count:
                count = 0
                count_x += 1
                last_x += 5.0 * (radius)

            
    if full_count > 0:
        last_x += 6.0 * radius + 1


l_x = last_x + 2.0*r_bins[0]
l_y =  max(x_lim)+6.0*r_bins[0]

fig = plt.figure(figsize=(12, 12*l_y/l_x),frameon=False, facecolor='k', edgecolor='k')
ax  = fig.add_axes([0,0,1,1], aspect='equal',axis_bgcolor='k', frameon=True)

for im in image_arr:
    ax.imshow( im[0], extent=im[1], interpolation='nearest' )

ax.set_xlim(-2*r_bins[0], last_x )
ax.set_ylim(-2*r_bins[0], max(x_lim)+4*r_bins[0])

ax.set_xticks([])
ax.set_yticks([])

plt.savefig('tidied_%s.png'%(fn.split('.')[0]), dpi=300)
plt.show()
