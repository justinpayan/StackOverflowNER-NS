entity_colors_so = {'Application': 'teal', 'Code_Block': 'mediumseagreen', 'Data_Structure': 'antiquewhite', 'Language': 'olivedrab', 'Library': 'beige', 'Library_Class': 'darkslategray', 'Library_Function': 'lightpink', 'User_Interface_Element': 'lightyellow', 'Value': 'forestgreen', 'Variable_Name': 'red', 'Other Entities': 'lime'}

def plot_colortable(color_map, emptycols=0):
    sort_colors = False
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40
    names = sorted(set(color_map.keys()) - {"Other Entities"})
    names.append("Other Entities")
    n = len(names)
    ncols = 1 - emptycols
    nrows = n // ncols + int(n % ncols > 0)
    width = cell_width * 1 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 1)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height
        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7
        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')
        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=color_map[name], linewidth=18)
    # plt.savefig("/Users/juspayan/plots/music_intent_pie_colors.png")
    plt.show()