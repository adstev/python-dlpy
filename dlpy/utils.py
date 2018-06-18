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
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# do k-means
def do_kmeans( boxes, centroids, n_anchors = 5):

    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss

def get_anchors(casTable, coordType, grid_size = 13, n_anchors = 5, loss_convergence = 1e-5):
    # reading all of boxes
    boxes = []

    # get only columns containing box width height and nObject info
    keep_cols = []
    for colName in casTable.columns:
        if any(s in colName for s in ['width', 'height', '_nObjects_']):
            keep_cols.append(colName)
    anchor_tbl = casTable.retrieve('table.partition', _messagelevel='error',
                            table=dict(Vars=keep_cols, **casTable.to_table_params()),
                            casout=dict(name='anchor_tbl', replace=True, blocksize=32))['casTable']

    # remove data points with no objects
    anchor_tbl = anchor_tbl[~anchor_tbl['_nObjects_'].isNull()]

    df = anchor_tbl.to_frame()
    # df = df[~df['_nObjects_'].isnull()]
    for idx, row in df.iterrows():
        n_object = int(row['_nObjects_'])
        for i in range(n_object):
            if coordType.lower() == 'yolo':
                width = float(row['_Object{}_width'.format(i)])
                height = float(row['_Object{}_height'.format(i)])
            elif coordType.lower() == 'rect':
                try:
                    img_width = int(row['_width_'])
                    img_height = int(row['_height_'])
                except:
                    print('Error: _width_ and _height_ columns should be in the table')
                    return
                width = float(row['_Object{}_width'.format(i)] / img_width)
                height = float(row['_Object{}_height'.format(i)] / img_height)
            else:
                print('Error: Only support Yolo and Rect coordType by far')
                return
            boxes.append(Box(0, 0, width, height))
    # initial centroids
    centroid_indices = np.random.choice(len(boxes), n_anchors)
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(boxes[centroid_index])

    # iterate k-means
    new_centroids, groups, old_loss = do_kmeans(boxes, centroids, n_anchors)

    while(True):
        new_centroids, groups, loss = do_kmeans(boxes, new_centroids, n_anchors)
        if abs(old_loss - loss) < loss_convergence:
            break
        old_loss = loss
    anchors = []
    for centroid in centroids:
        anchors += [centroid.w * grid_size, centroid.h * grid_size]
    return tuple(anchors)


# def get_max_objects(conn, casTable):
#     if isinstance(casTable, string):
#         table = conn.CASTable(casTable)
#     elif isinstance(casTable, CASTable):
#         table = casTable
#     else:
#         raise ValueError('Input table not valid name or casTable')
#     table = conn.CASTable(table)
#     summary = table.summary()['Summary']
#     return summary[summary['Column'] == '_nObjects_']['Max'].values[0]
#
#
# def show_with_object_labels(conn, casTable, coordType='RECT', nimages=5, ncol=8, randomize=False, figsize=None):
#     if isinstance(casTable, string):
#         table = conn.CASTable(casTable)
#     elif isinstance(casTable, CASTable):
#         table = casTable
#     else:
#         raise ValueError('Input table not valid name or casTable')
#
#     conn._retrive_('image.extractDetectedObjects',
#                    casout={'name': 'dataAnnotedresize', 'replace': True},
#                    coordType=coordType, maxobjects=get_max_objects(table), table=table)
#     # r = self.conn.image.fetchImages(table={'name': 'dataAnnotedresize'})
#     # r.Images['Image'][0]
#     from .images import ImageTable
#     ImageTable.from_table('dataAnnotedresize').show(nimages, ncol, randomize, figsize)
