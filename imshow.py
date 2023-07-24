import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import colorbar

plt.style.use("classic")
rc('axes',edgecolor='k')

nice_fonts = {"text.usetex": True,"font.family": "serif"}
mpl.rcParams.update(nice_fonts)

########################################################################################################
### Figure canvas tuning ###

fig_left_margin=0.1
fig_right_margin=0.1
fig_bottom_margin=0.1
fig_top_margin=0.1
#######################

### subplots tuning ###
rows=1
cols=1

horizontal_space=0.01
verticle_space=0.01
#######################

### inset tuning ###
inset_left_margin=0.1
inset_bottom_margin=0.5
inset_width=0.4    # out of 1.0
inset_height=0.4   # out of 1.0
#######################

### multiplot initialization ###

#fig = plt.figure(figsize=(10,10))

#### If aspect ratio is equal #######
fig = plt.figure(figsize=(5*cols,5*rows))
verticle_space=horizontal_space*cols/rows
#####################################

width = (1.0-(fig_left_margin+fig_right_margin)-(cols-1)*horizontal_space)/cols +0.12
height = (1.0-(fig_bottom_margin+fig_top_margin)-(rows-1)*verticle_space)/rows

main_ax=[]
for i in range(rows):
    tmp_ax=[]
    for j in range(cols):
        tmp_ax.append(plt.axes([ fig_left_margin + j*(horizontal_space+width),
                                 fig_bottom_margin + i*(verticle_space+height),
                                 width,
                                 height]))
    main_ax.append(tmp_ax)
#######################

### inset initialization ###

#inset_ax=[]
#for i in range(rows):
    #tmp_ax=[]
    #for j in range(cols):
        #tmp_ax.append(plt.axes([ fig_left_margin + j*(horizontal_space+width) + inset_left_margin*width,
                                 #fig_bottom_margin + i*(verticle_space+height) + inset_bottom_margin*height,
                                 #inset_width*width,
                                 #inset_height*height]))
    #inset_ax.append(tmp_ax)
#######################
########################################################################################################

separation=5.5

#datafile = ['xrange_7.5-31.5_yrange_11.5-35.5_431.dat',
            #'xrange8.5-32.5_yrange_9.5-33.5_429.dat',
            #]

datafile = ['Lambda_k_band_spin0_bands0_0Version1.txt']

nx =  [5184]
ny =  [5184]
#cx1 = [6.5,7.5,10]
#cx2 = [30.5,31.5,34]
#cy1 = [10.5,8.5,10]
#cy2 = [34.5,32.5,34]

#cx1 = [10,7.5,6.5]
#cx2 = [34,31.5,30.5]
#cy1 = [10,8.5,10.5]
#cy2 = [34,32.5,34.5]

#text_x=[separation*6.0*0.37-separation,
        #separation*6.49*0.37-separation,
        #separation*7.7*0.37-separation]

#text_y=[separation*7.19*0.965-separation,
        #separation*6.8*0.965-separation,
        #separation*7.1*0.965-separation]

#text_x=[separation*7.7*0.37-separation,
        #separation*6.49*0.37-separation,
        #separation*6.0*0.37-separation]

#text_y=[separation*7.105*0.965-separation,
        #separation*6.82*0.965-separation,
        #separation*7.198*0.965-separation]

#plot_marker = [r'(a)',r'(b)',r'(c)']

#ctiks = []
#climit_low = [-0.05,-0.08,-0.06]
#climit_up =  [ 0.05, 0.03, 0.01]

#climit_low = [-0.06,-0.08,-0.05]
#climit_up =  [ 0.01, 0.03, 0.05]

color_scheme = 'gnuplot'            # choice: gnuplot, viridis, hsv, winter, magma, inferno

for i in range(rows):
    for j in range(cols):
        ii = (i)*cols + j
        
        (sx,sy,sz)=np.genfromtxt(datafile[ii],usecols=(0,1,2),unpack=True)
        #print(i,j,ii,datafile[ii],plot_marker[ii])
        sx=np.reshape(sx,(nx[ii],ny[ii]),order='F')
        sy=np.reshape(sy,(nx[ii],ny[ii]),order='F')
        sz=np.reshape(sz,(nx[ii],ny[ii]),order='F')

        #main_ax[i][j].set_xlim(0,nx)
        #main_ax[i][j].set_ylim(0,ny)
        main_ax[i][j].set_xticks([])
        main_ax[i][j].set_yticks([])
        
        main_ax[i][j].spines['right'].set_linewidth(1.5)
        main_ax[i][j].spines['top'].set_linewidth(1.5)
        main_ax[i][j].spines['left'].set_linewidth(1.5)
        main_ax[i][j].spines['bottom'].set_linewidth(1.5)

        norm = mpl.colors.Normalize(vmin=-1.0,vmax=1.0)
        cmap = plt.get_cmap(color_scheme)
        
        Z1 = np.zeros((nx[ii],ny[ii]))
        for iy in range (ny[ii]):
          for ix in range (nx[ii]):
            jy = ix
            jx = ny[ii] - iy - 1
            Z1[jx,jy] = sz[ix,iy]
        
        im = main_ax[i][j].imshow(Z1,cmap=cmap,interpolation='hermite', alpha=1.0)

        #bbox={'facecolor': 'black', 'pad': 2.0}
        #main_ax[i][j].text(text_x[ii],text_y[ii], plot_marker[ii], color='white',fontsize=25, bbox= bbox)

        ax_divider = make_axes_locatable(main_ax[i][j])
        # add an axes to the right of the main axes.
        cax = ax_divider.append_axes("right", size="5%", pad="3%")

        cax.spines['right'].set_linewidth(0.005)
        cax.spines['right'].set_linewidth(0.005)
        cax.spines['top'].set_linewidth(0.005)
        cax.spines['left'].set_linewidth(0.005)
        cax.spines['bottom'].set_linewidth(0.005)

        cbar = fig.colorbar(im, cax=cax)
        norm = mpl.colors.Normalize(vmin=np.min(sz),vmax=np.max(sz))
        cmap = plt.get_cmap(color_scheme)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cax.tick_params(labelsize=25)
        cb = plt.colorbar(sm,cax=cax)


plt.savefig('2try.pdf',format='pdf',bbox_inches='tight',dpi=100)
#plt.show()
