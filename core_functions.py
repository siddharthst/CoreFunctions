def TPMNorm(fragLength, dataFrame, lengthDict):
    """
    TPM normalisation function - it takes three arguments.
    fragLenght = Length of the fragment.
    dataFrame = Count matrix in form of pandas dataframe
    lengthDict = Dictionary containing length of each gene
    in form of gene:geneLengthinBasePairs 
    """
    #----------Start imports----------#
    import numpy as np
    import pandas as pd
    #-----------End imports-----------#

    count = dataFrame
    sumCount = count.sum()
    sumCount = sumCount.tolist()
    count = count * (fragLength * 1000000)
    length = count.index.to_series().map(lengthDict)
    length = length.tolist()
    scaleMatrix = np.zeros(shape=(len(sumCount), len(length)))
    for i in range(0, len(sumCount)):
        for k in range(0, len(length)):
            scaleMatrix[i, k] = sumCount[i] * length[k]
    scaleDataframe = pd.DataFrame(
        data=scaleMatrix.T, columns=count.columns, index=count.index)
    count = count.divide(scaleDataframe)
    count = count + 1
    count = np.log2(count)
    return count


def pcaPlotSea(comp,
               dataFrame,
               filename,
               x_inv=False,
               y_inv=False,
               legend=False):
    '''
    PCA plotting function. It takes dataframe as input along with
    number of componenets to analyze. The output will always be a 
    scatter plot between first two components. It also accepts the
    filename for output. The optional two parameters (x_inv, y_inv) 
    define if any of the axis needs to be inverted or not.
    '''
    #----------Start imports----------#
    from sklearn.decomposition import PCA
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import seaborn as sns
    rcParams.update({'figure.autolayout': False})
    sns.set(style="ticks")
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams["font.family"] = "DIN Alternate"
    rcParams['axes.linewidth'] = 5
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    plt.rc("font", family="DIN Alternate")
    rcParams['axes.linewidth'] = 5
    rcParams['axes.labelsize'] = 36
    #-----------End imports-----------#

    count = dataFrame
    n = comp
    palette = sns.color_palette("tab20c", n_colors=17)
    pca = PCA(n_components=n)
    count_t = count.T
    x = pca.fit(count_t)
    columns = ['pca_%i' % i for i in range(n)]
    df_pca = pd.DataFrame(
        x.transform(count_t), columns=columns, index=count_t.index)
    df_pca['Day'] = df_pca.index.astype('category')
    size = pca.explained_variance_ratio_
    plt.figure()
    f, ax = plt.subplots(figsize=(12, 12))
    ax = sns.scatterplot(
        x="pca_0",
        y="pca_1",
        hue="Day",
        sizes=(10, 200),
        data=df_pca,
        legend=legend,
        palette=palette,
        s=400,
        linewidth=0)
    sns.despine(f, right=True, top=True, offset=10, trim=True)
    ax.set(
        xlabel="{:.1f}".format(size[0] * 100) + "% variance",
        ylabel="{:.1f}".format(size[1] * 100) + "% variance")
    ax.tick_params(direction='out', length=10, width=5)
    my_xticks = ax.get_xticks()
    plt.xticks(
        [my_xticks[0], 0.0, my_xticks[-1]],
        visible=True,
        rotation="horizontal")
    my_yticks = ax.get_yticks()
    plt.yticks(
        [my_yticks[0], 0.0, my_yticks[-1]],
        visible=True,
        rotation="horizontal")
    plt.tick_params(labelsize=40)
    if (x_inv == True):
        ax.invert_xaxis()
    if (y_inv == True):
        ax.invert_yaxis()
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(filename)


def plotClusterMap(dataFrame,
                   filename,
                   supressRowDend=False,
                   supressColDend=False,
                   clusterColumn=True,
                   clusterRow=True,
                   vUpper=10.0,
                   colorColumn=0):
    '''
    Function to create clustermap based on seaborn clustermap
    function. It takes dataframe containing counts with index 
    as gene names and filename for the output. It also allows
    supression of row and column dendogram without affecting 
    clustering using parameters supressRowDend/supressColDend.
    '''
    #----------Start imports----------#
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    #-----------End imports-----------#

    #---------Submodule start---------#
    def aggloCluster(numberColumn, matrix):
        '''
        Subfunction which accepts number of clusters and
        dataframe. It returns colored labels which can 
        be used with clustermap
        '''
        #Performing clustering at this step to identify groups for different
        #stages
        model = AgglomerativeClustering(
            n_clusters=numberColumn, linkage="average")
        model.fit(matrix)
        labels = (model.labels_).tolist()
        return labels

    def SaveFigure(ClusterMap,
                   supressRowDend=supressRowDend,
                   supressColDend=supressColDend,
                   filename=filename,
                   vUpper=vUpper):
        if (supressRowDend == True):
            ClusterMap.ax_row_dendrogram.set_visible(False)
        if (supressColDend == True):
            ClusterMap.ax_col_dendrogram.set_visible(False)
        fig = ClusterMap.savefig(filename)
        #Create the colorbar
        plt.figure()
        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=0, vmax=vUpper)
        cb1 = mpl.colorbar.ColorbarBase(
            ax1, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Log2 expression value')
        cb1.ax.tick_params(labelsize=30)
        cb1.outline.set_visible(False)
        fig.savefig('plots/Figure1/heatmap/colorbar.pdf')

    #----------Submodule end----------#

    plt.figure()
    rowLen = len(dataFrame.index)

    #Create the clustermap
    if (colorColumn > 0):
        #Creating colored labels for groups created for columns
        labels = aggloCluster(colorColumn, dataFrame.T.values)
        columnNames = list(dataFrame.columns.values)
        labels = [
            i
            for x, y in zip(
                sns.color_palette("Set2", len(set(labels))),
                sorted(set(labels), key=labels.index))
            for i in np.repeat([x], labels.count(y), axis=0).tolist()
        ]
        group_lut = dict(zip(map(int, columnNames), labels))
        Column_colors = pd.Series(group_lut)

        ClusterMap = sns.clustermap(
            dataFrame,
            cmap="inferno",
            linewidths=0,
            figsize=(6, rowLen / 8),
            robust=True,
            col_cluster=clusterColumn,
            row_cluster=True,
            vmin=np.log2(1),
            vmax=vUpper,
            col_colors=Column_colors,
            yticklabels=False)
        ClusterMap.cax.set_visible(False)
        ax = ClusterMap.ax_heatmap
        ax.set_ylabel("")
        SaveFigure(ClusterMap)

    else:
        ClusterMap = sns.clustermap(
            dataFrame,
            cmap="inferno",
            linewidths=0,
            figsize=(6, rowLen / 8),
            robust=True,
            col_cluster=clusterColumn,
            row_cluster=True,
            vmin=np.log2(1),
            vmax=vUpper,
            yticklabels=False)
        ClusterMap.cax.set_visible(False)
        ax = ClusterMap.ax_heatmap
        ax.set_ylabel("")
        SaveFigure(ClusterMap)


def plotClusterMapBi(dataFrame,
                     filename,
                     supressRowDend=False,
                     supressColDend=False,
                     vUpper=10.0):
    '''
    This needs refactoring too. Decided to create a seperate function
    for bidirectional genes to mark the pair and determine which arm
    is positive and negative. This is essentially the same cluster 
    plot but with bidirectional genes marked :)
    '''
    #----------Start imports----------#
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    import itertools
    import matplotlib
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': False})
    sns.set(style="ticks")
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams["font.family"] = "DIN Alternate"
    rcParams['axes.linewidth'] = 5
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="DIN Alternate")

    #-----------End imports-----------#

    #---------Submodule start---------#
    def SaveFigure(ClusterMap,
                   supressRowDend=supressRowDend,
                   supressColDend=supressColDend,
                   filename=filename,
                   vUpper=vUpper):
        if (supressRowDend == True):
            ClusterMap.ax_row_dendrogram.set_visible(False)
        if (supressColDend == True):
            ClusterMap.ax_col_dendrogram.set_visible(False)
        fig = ClusterMap.savefig(filename)

    #----------Submodule end----------#

    plt.figure()
    rowLen = len(dataFrame.index)

    #Create the clustermap
    colorRow = 1
    if (colorRow > 0):
        #Creating colored labels for groups created for columns
        listRange = list(range(1, int(rowLen / 2) + 1))
        labels = [x for pair in zip(listRange, listRange) for x in pair]
        dataFrame.index = list(range(1, int(rowLen) + 1))
        rowNames = list(dataFrame.index.values)
        labels = [
            i
            for x, y in zip(
                sns.color_palette("tab20b", len(set(labels))),
                sorted(set(labels), key=labels.index))
            for i in np.repeat([x], labels.count(y), axis=0).tolist()
        ]
        group_lut = dict(zip(map(int, rowNames), labels))
        Row_colors = pd.Series(group_lut)
        ylabels = [
            x
            for x in itertools.chain.from_iterable(
                itertools.zip_longest(['—'] * int(rowLen / 2),
                                      ['+'] * int(rowLen / 2))) if x
        ]
        ClusterMap = sns.clustermap(
            dataFrame,
            cmap="inferno",
            linewidths=0,
            figsize=(6, rowLen / 8),
            robust=True,
            col_cluster=False,
            row_cluster=False,
            vmin=np.log2(1),
            vmax=vUpper,
            row_colors=Row_colors,
            yticklabels=ylabels)
        ClusterMap.cax.set_visible(False)
        ax = ClusterMap.ax_heatmap
        ax.set_ylabel("")
        ax.tick_params(axis=u'both', which=u'both', length=0)
        SaveFigure(ClusterMap)

    else:
        ClusterMap = sns.clustermap(
            dataFrame,
            cmap="inferno",
            linewidths=0,
            figsize=(6, rowLen / 8),
            robust=True,
            col_cluster=False,
            row_cluster=False,
            vmin=np.log2(1),
            vmax=vUpper,
            yticklabels=False)
        ClusterMap.cax.set_visible(False)
        ax = ClusterMap.ax_heatmap
        ax.set_ylabel("")
        SaveFigure(ClusterMap)


def kmeans(dataframe, numberOfClusters=2, random_state=10):
    '''
    Kmeans clustering function: It takes dataframes as input 
    and returns a dataframe with clusters generated using
    kmeans algorithm in scikit. Option argument includes number 
    of required clusters(default = 2) and random state.
    '''
    #----------Start imports----------#
    from sklearn.cluster import KMeans
    import pandas as pd
    #-----------End imports-----------#

    mat = dataframe.values
    km = KMeans(n_clusters=numberOfClusters)
    km.fit(mat)
    labels = km.labels_
    results = pd.DataFrame([dataframe.index, labels]).T
    return (results)


class renamer:
    '''
    Function - or class ? For renaming dupliate columns.
    https://stackoverflow.com/questions/40774787/renaming-columns-in-a-pandas-dataframe-with-duplicate-column-names?rq=1
    '''

    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s.%d" % (x, self.d[x])


def plotClusterMapUnscaled(dataFrame,
                           filename,
                           supressRowDend=False,
                           supressColDend=False,
                           clusterColumn=True,
                           clusterRow=True,
                           vUpper=10.0,
                           colorColumn=0,
                           l=10,
                           w=10):
    '''
    Function to create clustermap based on seaborn clustermap
    function. It takes dataframe containing counts with index 
    as gene names and filename for the output. It also allows
    supression of row and column dendogram without affecting 
    clustering using parameters supressRowDend/supressColDend.
    The difference with other similarly names function is the 
    fact that this one does not scaled anything and instead 
    accepts the dimensions.
    '''
    #----------Start imports----------#
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    #-----------End imports-----------#

    #---------Submodule start---------#
    def aggloCluster(numberColumn, matrix):
        '''
        Subfunction which accepts number of clusters and
        dataframe. It returns colored labels which can 
        be used with clustermap
        '''
        #Performing clustering at this step to identify groups for different
        #stages
        model = AgglomerativeClustering(
            n_clusters=numberColumn, linkage="average")
        model.fit(matrix)
        labels = (model.labels_).tolist()
        return labels

    def SaveFigure(ClusterMap,
                   supressRowDend=supressRowDend,
                   supressColDend=supressColDend,
                   filename=filename,
                   vUpper=vUpper):
        if (supressRowDend == True):
            ClusterMap.ax_row_dendrogram.set_visible(False)
        if (supressColDend == True):
            ClusterMap.ax_col_dendrogram.set_visible(False)
        fig = ClusterMap.savefig(filename)
        #Create the colorbar
        plt.figure()
        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=0, vmax=vUpper)
        cb1 = mpl.colorbar.ColorbarBase(
            ax1, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Log2 expression value')
        cb1.ax.tick_params(labelsize=30)
        cb1.outline.set_visible(False)
        fig.savefig('plots/Figure1/heatmap/colorbar.pdf')

    #----------Submodule end----------#

    plt.figure()
    rowLen = len(dataFrame.index)

    #Create the clustermap
    if (colorColumn > 0):
        #Creating colored labels for groups created for columns
        labels = aggloCluster(colorColumn, dataFrame.T.values)
        columnNames = list(dataFrame.columns.values)
        labels = [
            i
            for x, y in zip(
                sns.color_palette("Set2", len(set(labels))),
                sorted(set(labels), key=labels.index))
            for i in np.repeat([x], labels.count(y), axis=0).tolist()
        ]
        group_lut = dict(zip(map(int, columnNames), labels))
        Column_colors = pd.Series(group_lut)

        ClusterMap = sns.clustermap(
            dataFrame,
            cmap="inferno",
            linewidths=0,
            figsize=(l, w),
            robust=True,
            col_cluster=clusterColumn,
            row_cluster=True,
            vmin=np.log2(1),
            vmax=vUpper,
            col_colors=Column_colors,
            yticklabels=False)
        ClusterMap.cax.set_visible(False)
        ax = ClusterMap.ax_heatmap
        ax.set_ylabel("")
        SaveFigure(ClusterMap)

    else:
        ClusterMap = sns.clustermap(
            dataFrame,
            cmap="inferno",
            linewidths=0,
            figsize=(6, rowLen / 8),
            robust=True,
            col_cluster=clusterColumn,
            row_cluster=True,
            vmin=np.log2(1),
            vmax=vUpper,
            yticklabels=False)
        ClusterMap.cax.set_visible(False)
        ax = ClusterMap.ax_heatmap
        ax.set_ylabel("")
        SaveFigure(ClusterMap)


def dfVariance(dataFrame, retSeries=False, type="row"):
    '''
    Function which returns the variance of 
    pandas dataframe on rows(default) or
    column. It also accepts arguments retSeries
    which if set True returns a series instead
    of dataframe with added column/row.
    '''
    #----------Start imports----------#
    import pandas as pd
    import numpy as np
    #-----------End imports-----------#
    if (type == "row"):
        if (retSeries == True):
            return (dataFrame.var(axis=1))
        dataFrame["rowVariance"] = dataFrame.var(axis=1)
        return (dataFrame)
    if (type == "column"):
        if (retSeries == True):
            return (dataFrame.var(axis=0))
        dataFrame.loc['colVariance'] = dataFrame.var(axis=0)
        return (dataFrame)
    if (type == "both"):
        dataFrame["rowVariance"] = dataFrame.var(axis=1)
        dataFrame.loc['colVariance'] = dataFrame.var(axis=0)
        return (dataFrame)


def plotFeatureDistance(dataframe, cutOffLine=400):
    '''
    This functions plots the distance between gene start
    and upstream gene.
    '''
