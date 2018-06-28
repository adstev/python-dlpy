#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

''' Utility functions for the DLPy package '''

import os
import random
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import six
import swat as sw
from swat.cas.table import CASTable


def random_name(name='ImageData', length=6):
    '''
    Generate random name

    Parameters
    ----------
    name : string, optional
        Prefix of the generated name
    length : int, optional
        Length of the random characters in the name

    Returns
    -------
    string

    '''
    return name + '_' + ''.join(random.sample(
        string.ascii_uppercase + string.ascii_lowercase + string.digits, length))


def input_table_check(input_table):
    '''
    Unify the input_table format

    Parameters
    ----------
    input_table : CASTable or string or dict
        Input table specification

    Returns
    -------
    dict
        Input table parameters

    '''
    if isinstance(input_table, six.string_types):
        input_table = dict(name=input_table)
    elif isinstance(input_table, dict):
        input_table = input_table
    elif isinstance(input_table, CASTable):
        input_table = input_table.to_table_params()
    else:
        raise TypeError('input_table must be one of the following:\n'
                        '1. A CAS table object;\n'
                        '2. A string specifies the name of the CAS table,\n'
                        '3. A dictionary specifies the CAS table\n'
                        '4. An ImageTable object.')
    return input_table


def prod_without_none(array):
    '''
    Compute the product of an iterable array with None as its element

    Parameters
    ----------
    array : iterable-of-numeric
        The numbers to use as input

    Returns
    -------
    numeric
        Product of all the elements of the array

    '''
    prod = 1
    for i in array:
        if i is not None:
            prod *= i
    return prod


def get_max_size(start_path='.'):
    '''
    Get the max size of files in a folder including sub-folders

    Parameters
    ----------
    start_path : string, optional
        The directory to start the file search

    Returns
    -------
    int

    '''
    max_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            file_size = os.path.getsize(fp)
            if file_size > max_size:
                max_size = file_size
    return max_size


def image_blocksize(width, height):
    '''
    Determine blocksize according to imagesize in the table

    Parameters
    ----------
    width : int
        The width of the image
    height : int
        The height of the image

    Returns
    -------
    int

    '''
    return width * height * 3 / 1024


def predicted_prob_barplot(ax, labels, values):
    '''
    Generate a horizontal barplot for the predict probability

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot to
    labels : list-of-strings
        Predicted class labels
    values : list-of-numeric
        Predicted probabilities

    Returns
    -------
    :class:`matplotlib.axes.Axes`

    '''
    y_pos = (0.2 + np.arange(len(labels))) / (1 + len(labels))
    width = 0.8 / (1 + len(labels))
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    for i in range(len(labels)):
        ax.barh(y_pos[i], values[i], width, align='center',
                color=colors[i], ecolor='black')
        ax.text(values[i] + 0.01, y_pos[i], '{:.2%}'.format(values[i]))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, rotation=45)
    ax.set_xlabel('Probability')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title('Predicted Probability')
    return ax


def plot_predict_res(image, label, labels, values):
    '''
    Generate a side by side plot of the predicted result

    Parameters
    ----------
    image :
        Specifies the orginal image to be classified.
    label : string
        Specifies the class name of the image.
    labels : list-of-strings
        Predicted class labels
    values : list-of-numeric
        Predicted probabilities

    Returns
    -------
    :class:`matplotlib.axes.Axes`

    '''
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('{}'.format(label))
    ax1.imshow(image)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    predicted_prob_barplot(ax2, labels, values)


def camelcase_to_underscore(strings):
    ''' Convert camelcase to underscore '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', strings)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def add_caslib(conn, path):
    '''
    Add a new caslib, as needed

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    path : string
        Specifies the server-side path to check

    Returns
    -------
    string
        The name of the caslib pointing to the path

    '''
    if path in conn.caslibinfo().CASLibInfo.Path.tolist():
        cas_lib_name = conn.caslibinfo().CASLibInfo[
            conn.caslibinfo().CASLibInfo.Path == path]['Name']

        return cas_lib_name.tolist()[0]
    else:
        cas_lib_name = random_name('Caslib', 6)
        conn.retrieve('table.addcaslib', message_level='error',
                      name=cas_lib_name, path=path, activeOnAdd=False,
                      dataSource=dict(srcType='DNFS'))
        return cas_lib_name


def upload_astore(conn, path, table_name=None):
    '''
    Load the local astore file to server

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    path : string
        Specifies the client-side path of the astore file
    table_name : string, or casout options
        Specifies the name of the cas table on server to put the astore object

    '''
    conn.loadactionset('astore')

    with open(path, 'br') as f:
        astore_byte = f.read()

    store_ = sw.blob(astore_byte)

    if table_name is None:
        table_name = random_name('ASTORE')
    conn.astore.upload(rstore=table_name, store=store_)


def unify_keys(dic):
    '''
    Change all the key names in a dictionary to lower case, remove "_" in the key names.

    Parameters
    ----------
    dic : dict

    Returns
    -------
    dict
        dictionary with updated key names

    '''

    old_names = list(dic.keys())
    new_names = [item.lower().replace('_', '') for item in old_names]
    for new_name, old_name in zip(new_names, old_names):
        dic[new_name] = dic.pop(old_name)

    return dic


def check_caslib(conn, path):
    '''
    Check whether the specified path is in the caslibs of the current session.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object

    path : str
        Specifies the name of the path.

    Returns
    -------
    flag : bool
        Specifies if path exist in session.
    caslib_name : str (if exist)
        Specifies the name of the caslib that contain the path.

    '''
    paths = conn.caslibinfo().CASLibInfo.Path.tolist()
    caslibs = conn.caslibinfo().CASLibInfo.Name.tolist()

    if path in paths:
        caslibname = caslibs[paths.index(path)]
        return True, caslibname
    else:
        return False


class Box():
    '''
    Box class used in object detection

    Parameters
    ----------
    x : float
        x location of box between 0 and 1 relative to image column position
    y : float
        y location of box between 0 and 1 relative to image row position
    w : float
        width of the box
    h : float
        height of the box

    Attributes
    ----------
    x : float
        x location of box between 0 and 1 relative to image column position
    y : float
        y location of box between 0 and 1 relative to image row position
    w : float
        width of the box between 0 and 1 relative to image width
    h : float
        height of the box between 0 and 1 relative to image height

    '''
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def get_iou_distance(box, centroid):
    '''
    Gets one minus the area intersection over union of two boxes

    Parameters
    ----------
    box : Box
        A Box object containing width and height information
    centroid : Box
        A Box object containing width and height information

    Returns
    -------
    float
        One minus intersection over union
        Smaller the number the closer the boxes
    '''
    w = min(box.w, centroid.w)
    h = min(box.h, centroid.h)
    intersection = w*h
    union = box.w * box.h + centroid.w * centroid.h - intersection
    iou = intersection/union
    return 1-iou


def run_kmeans( boxes, centroids, n_anchors = 5):
    '''
    Runs a single iteration of the k-means algorithm

    Parameters
    ----------
    boxes : list
        Contains Box objects with width and height labels from training data
    centroids : list
        List of boxes containing current width and height info
        for each cluster cluster center
    n_anchors : int
        Number of anchors for each grid cell

    Returns
    -------
    list
        new_centroids : updated list of Box objects containing width and height
        of each anchor box
    list
        clusters : List of list of Box objects grouped by which centroid they
        are closest to.
    float
        loss : sum of distances of each Box in training data to its closest centroid
    '''
    loss = 0
    clusters = []
    new_centroids = []
    for i in range(n_anchors):
        clusters.append([])
        new_centroids.append(Box(0, 0, 0, 0))
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = get_iou_distance(box, centroid)
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        clusters[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(clusters[i])
        new_centroids[i].h /= len(clusters[i])

    return new_centroids, clusters, loss

def get_anchors(cas_table, coord_type='yolo', grid_number = 13, n_anchors = 5, loss_convergence = 1e-5):
    '''
    Gets best n_anchors for object detection grid cells based on k-means.

    Parameters
    ----------
    cas_table : CASTable
        Specifies the table containing object detection box labels
    coord_type : string
        Specifies the format of the box labels
        Can be 'yolo', or 'rect'
        'yolo' specifies x, y, width and height, x, y is the center
        location of the object in the grid cell. x, y, are between 0
        and 1 and are relative to that grid cell. x, y = 0,0 corresponds
        to the top left pixel of the grid cell.
    grid_number : int
        The number of grids in each row or column.
        Total number of grids = grid_number*grid_number
    n_anchors : int
        Number of anchors in each grid cell
    loss_convergence : float
        If the change in k-means loss from one iteration to the next
        is smaller than loss_convergence then k-means has converged

    Returns
    -------
    tuple
        Contains widths and heights of anchor boxes

    '''
    boxes = []
    # get only columns containing box width height and nObject info
    keep_cols = []
    for col_name in cas_table.columns:
        if any(s in col_name for s in ['width', 'height', '_nObjects_']):
            keep_cols.append(col_name)
    anchor_tbl = cas_table.retrieve('table.partition', _messagelevel='error',
                                    table=dict(Vars=keep_cols, **cas_table.to_table_params()),
                                    casout=dict(name='anchor_tbl', replace=True, blocksize=32))['casTable']

    # remove data points with no objects
    anchor_tbl = anchor_tbl[~anchor_tbl['_nObjects_'].isNull()]

    df = anchor_tbl.to_frame()
    # df = df[~df['_nObjects_'].isnull()]
    for idx, row in df.iterrows():
        n_object = int(row['_nObjects_'])
        for i in range(n_object):
            if coord_type.lower() == 'yolo':
                width = float(row['_Object{}_width'.format(i)])
                height = float(row['_Object{}_height'.format(i)])
            elif coord_type.lower() == 'rect':
                try:
                    img_width = int(row['_width_'])
                    img_height = int(row['_height_'])
                except:
                    print('Error: _width_ and _height_ columns should be in the table')
                    return
                width = float(row['_Object{}_width'.format(i)] / img_width)
                height = float(row['_Object{}_height'.format(i)] / img_height)
            else:
                print('Error: Only support Yolo and Rect coordType so far')
                return
            boxes.append(Box(0, 0, width, height))
    # Randomly assign centroids
    centroid_indices = np.random.choice(len(boxes), n_anchors)
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(boxes[centroid_index])

    centroids, clusters, old_loss = run_kmeans(boxes, centroids, n_anchors)

    while(True):
        centroids, clusters, loss = run_kmeans(boxes, centroids, n_anchors)
        if abs(old_loss - loss) < loss_convergence:
            break
        old_loss = loss
    anchors = []
    for centroid in centroids:
        anchors += [centroid.w * grid_number, centroid.h * grid_number]
    return tuple(anchors)


def get_max_objects(cas_table):
    '''
    Get the maximum number of objects in an image from all instances in dataset

    Parameters
    ----------
    cas_table : CASTable
        Specifies the table containing the object detection data.
        Must contain column '_nObjects_'

    Returns
    -------
    int
        Maximum number of objects found in an image in the dataset

    '''

    if isinstance(cas_table, CASTable):
        pass
    else:
        raise ValueError('Input table not valid name or CASTable')
    if '_nObjects_' not in cas_table.columns.tolist():
        raise ValueError('Input table must contain _nObjects_ column')
    if (cas_table['_nObjects_'] < 1).all():
        raise ValueError('CASTable {} contains no images with labeled objects'.format(cas_table.name))

    summary = cas_table.summary()['Summary']
    return int(summary[summary['Column'] == '_nObjects_']['Max'].tolist()[0])
    # return int(cas_table.describe()['_nObjects_']['max'])


def filter_by_filename(cas_table, filename, filtered_name=None):
    '''
    Filteres CASTable by filename using '_path_' or '_filename_0' column

    Parameters
    ----------
    cas_table : CASTable
        Specifies the table to be filtered
        Note: CASTable should have a '_path_' or '_filename_0' column
        and an '_id_' column
    filename : string
        Can be part or full name of image or path
        If not unique, returns all that contain filename
    filtered_name : str
        Name of output table

    Returns
    -------
    CASTable
        Filtered table containing all instances that have

    '''
    if filtered_name:
        if isinstance(filtered_name, str):
            new_name = filtered_name
        else:
            raise ValueError('filtered_name must be a str or leave as None to generate a random name')
    else:
        new_name = random_name('filtered')

    if '_path_' not in cas_table.columns.tolist() and '_filename_0' not in cas_table.columns.tolist():
        raise ValueError('\'_path_\' or \'_filename_0\' column not found in CASTable : {}'.format(cas_table.name))
    if isinstance(filename, list):
        image_id = []
        for name in filename:
            if '_path_' in cas_table.columns.tolist():
                id_num = cas_table[cas_table['_path_'].str.contains(name)]['_id_'].tolist()
                if id_num:
                    image_id.extend(id_num)
            elif '_filename_0' in cas_table.columns.tolist():
                id_num = cas_table[cas_table['_filename_0'].str.contains(name)]['_id_'].tolist()
                if id_num:
                    image_id.extend(id_num)
    elif isinstance(filename, str):
        if '_path_' in cas_table.columns.tolist():
            image_id = cas_table[cas_table['_path_'].str.contains(filename)]['_id_'].tolist()
        elif '_filename_0' in cas_table.columns.tolist():
            image_id = cas_table[cas_table['_filename_0'].str.contains(filename)]['_id_'].tolist()

    if not image_id:
        raise ValueError('filename: {} not found in \'_path_\' or'
                         '\'_filename_0\' columns of table'.format(filename))

    return filter_by_image_id(cas_table, image_id, filtered_name=new_name)


def filter_by_image_id(cas_table, image_id, filtered_name=None):
    '''
    Filter CASTable by '_id_' column

    Parameters
    ----------
    cas_table : CASTable
        Specifies the table to be filtered
        Note: CASTable should have an '_id_' column
    image_id : int or list of ints
        Specifies the image id or ids to be kept
    filtered_name : str
        Name of output table

    Returns
    -------
    CASTable
        Filtered table by image_id

    '''
    if filtered_name:
        if isinstance(filtered_name, str):
            new_name = filtered_name
        else:
            raise ValueError('filtered_name must be a str or left as None to generate a random name')
    else:
        new_name = random_name('filtered')

    if '_id_' not in cas_table.columns.tolist():
        raise ValueError('\'_id_\' column not found in CASTable : {}'.format(cas_table.name))
    cas_table = cas_table[cas_table['_id_'].isin(image_id)]
    if cas_table.numrows().numrows == 0:
        raise ValueError('image_id: {} not found in the table'.format(image_id))

    filtered = cas_table.partition(casout=dict(name=new_name, replace=True))['casTable']

    return filtered


def check_annotated_images(conn, cas_table, coord_type='yolo', labeled_name=None, display=True,
                           return_table=True, image_id=None, filename=None, nimages=5, ncol=8,
                           randomize=False, figsize=None):
    '''
    Draw bounding boxes on images in cas_table
    Note: Images without any objects are removed

    Parameters
    ----------
    conn : CAS
        The CAS connection object containing images
    cas_table : CASTable or string
        Specifies the table containing the object detection data and images.
    coord_type : string
        Specifies the format of the box labels
        Can be 'yolo', or 'rect'
        'yolo' specifies x, y, width and height, x, y is the center
        location of the object in the grid cell. x, y, are between 0
        and 1 and are relative to that grid cell. x, y = 0,0 corresponds
        to the top left pixel of the grid cell.
    labeled_name : string
        name of output table containing annotated images
    display : boolean
        If True display images
    return_table : boolean
        If True return table with annotated images
    image_id : int or list of ints
        Specifies an '_id_' to display
    filename : string
        Name of file to dispalay
    nimages : int, optional
        Specifies the number of images to be displayed.
        If nimage is greater than the maximum number of images in the
        table, it will be set to this maximum number.
        Note: Specifying a large value for nimages can lead to slow
        performance.
    ncol : int, optional
        Specifies the layout of the display, determine the number of
        columns in the plots.
    randomize : boolean, optional
        Specifies whether to randomly choose the images for display.
    figsize : tuple of ints
        Size of matplotlib figure

    Returns
    ----------
    CASTable
        New cas_table with images annotated

    '''
    #     conn.loadactionset('image', _messagelevel='error')

    if not conn.queryactionset('image').image:
        conn.loadactionset('image', _messagelevel='error')

    if not isinstance(cas_table, CASTable):
        raise ValueError('Input table not valid name or cas_table')

    # create copy of table so can be dropped from caslib as needed
    cas_table = cas_table.partition(casout=dict(name='temp_anotated', replace=True))['casTable']

    # generate a random name if none provided
    if not labeled_name:
        out_name = random_name('labeled_img_tbl')
    elif not isinstance(labeled_name, str):
        raise ValueError('labeled_name must be a string')
    else:
        out_name = labeled_name

    filtered = None
    if filename or image_id:

        if '_id_' not in cas_table.columns.tolist():
            print("'_id_' column not in cas_table, processing complete table")
        else:
            if filename and image_id:
                print(" image_id supersedes filename, image_id being used")

            if image_id:
                filtered = filter_by_image_id(cas_table, image_id)
            elif filename:
                filtered = filter_by_filename(cas_table, filename)

        if filtered:
            conn.droptable(cas_table)
            cas_table = filtered

    if '_nObjects_' in cas_table.columns.tolist():
        print("Extracting labels onto images")
        objects = get_max_objects(cas_table)
        cas_table._retrieve('image.extractDetectedObjects', casout={'name': out_name, 'replace': True},
                            coordType=coord_type, maxobjects=objects, table=cas_table)

        anotated = conn.CASTable(out_name)
        conn.droptable(cas_table)

    else:
        conn.droptable(cas_table)
        raise ValueError(" cas_table must contain '_nObjects_' column ")

    if display:
        show_images(conn, anotated, n_images=nimages,
                    n_col=ncol, randomize=randomize, fig_size=figsize)
    if return_table:
        return anotated
    elif labeled_name:
        return None
    else:
        conn.droptable(anotated)
        return None


def show_images(conn, cas_table, image_id=None, filename=None, n_images=5, n_col=8, randomize=False, fig_size=None):
    '''
    Display a grid of images

    Parameters
    ----------
    conn : CAS
        The CAS connection object containing images
    cas_table : CASTable or string
        Specifies the table containing the images.
    image_id : int or list of ints
        Specifies an '_id_' to display
    filename : string
        Name of file to dispalay
    n_images : int, optional
        Specifies the number of images to be displayed.
        If nimage is greater than the maximum number of images in the
        table, it will be set to this maximum number.
        Note: Specifying a large value for nimages can lead to slow
        performance.
    n_col : int, optional
        Specifies the layout of the display, determine the number of
        columns in the plots.
    randomize : boolean, optional
        Specifies whether to randomly choose the images for display.
    fig_size : tuple of ints
        Size of matplotlib figure

    '''
    # import matplotlib.pyplot as plt
    # import numpy as np

    if not conn.queryactionset('image').image:
        conn.loadactionset('image', _messagelevel='error')

    if isinstance(cas_table, str):
        cas_table = conn.CASTable(cas_table)
    elif isinstance(cas_table, CASTable):
        pass
    else:
        raise ValueError('Input table not valid name or CASTable')

    data = cas_table.partition(casout=dict(name='just_for_show', replace=True))['casTable']

    if filename or image_id:
        if '_id_' not in data.columns.tolist():
            raise ValueError('\'_id_\' column required but not found in table')
    if filename and image_id:
        print(" image_id supersedes filename, image_id being used")
    elif image_id:
        print("Filtering by id")
        data_temp = filter_by_image_id(data, image_id)
        conn.droptable(data)
        data = data_temp
    elif filename:
        data_temp = filter_by_filename(data, filename)
        conn.droptable(data)
        data = data_temp

    n_images = min(n_images, len(data))



    if randomize:
        data.append_computedvars(['random_index'])
        data.append_computedvarsprogram('call streaminit(-1);' 'random_index=''rand("UNIFORM")')

        temp_tbl = data.retrieve('image.fetchimages', _messagelevel='error',
                                 table=dict(**data.to_table_params()),
                                 fetchVars=['_id_'],
                                 sortby='random_index', to=n_images)
    else:
        temp_tbl = data.retrieve('image.fetchimages', fetchVars=['_id_'], to=n_images,
                                 sortBy=[{'name': '_id_', 'order': 'ASCENDING'}])

    if n_images > n_col:
        nrow = n_images // n_col + 1
    else:
        nrow = 1
        n_col = n_images
    if fig_size is None:
        fig_size = (16, 16 // n_col * nrow)
    fig = plt.figure(figsize=fig_size)

    for i in range(n_images):
        image = temp_tbl['Images']['Image'][i]
        label = temp_tbl['Images']['_id_'][i]

        ax = fig.add_subplot(nrow, n_col, i + 1)
        ax.set_title('image_id: {}'.format(label))
        if len(image.size) == 2:
            plt.imshow(np.array(image), cmap='Greys_r')
        else:
            plt.imshow(image)
        plt.xticks([]), plt.yticks([])

    conn.droptable(data)