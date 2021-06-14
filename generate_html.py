import os
import sys
import numpy as np
import pandas as pd
import webbrowser

def get_color_name(i):
  colors = [
    'AliceBlue',
    'AntiqueWhite',
    'LightGoldenRodYellow',
    'PaleGoldenRod',
    'PeachPuff',
    'Snow',
  ]
  return colors[i % len(colors)]

def image_path_to_link(p):
  return f"<a href=\"{p}\" target=\"_blank\">Image</a>"

def info_table():
  keys = {
    'Failure': '1 if stereo failed OR if when projecting all estimated 3D points into the nearest (integer) pixel coordinate in the depth image, less than 10% of the image is "covered".',
    'Debug depth': 'A link to a "debug image" comparing ground truth vs. estimated depth. Coloring is based on z-coordinate in camera frame, NOT ray length from camera to surface, and is scaled by the overall (min, max) z-coordinate from estimated and ground truth points.',
    'Debug stereo': 'A link to a "debug image" from the stereo algorithm, this is specific and different for each algorithm.',
    'Fraction estimated px': 'Calculated by projecting all estimated 3D points into the nearest (integer) pixel coordinate in the depth image, and calculating how many pixels are filled',
    'Fraction inliers': 'The fraction of estimated points with an error below 6 cm. Based on estimated vs. true ray length from camera origin to surface.',
    'Fraction outliers': '1 - (fraction of inliers)',
    'G.T. relative error': 'Defined as (estimated depth - true depth) / true depth. Based on estimated vs. true ray length from camera origin to surface.',
    'Max abs. error': 'Maximum error in meters of any point estimate. Based on estimated vs. true ray length from camera origin to surface.',
    'Median abs. error': 'Median error in meters of any point estimate. Based on estimated vs. true ray length from camera origin to surface.',
    'Outliers / inliers': '(fraction of outlier px) / (fraction of inlier px), i.e. How many outliers for each inlier?',
    'RMSE': 'Root mean squared error of estimated vs. ground truth depth. Based on estimated vs. true ray length from camera origin to surface.'
  }
  index = list(sorted(keys.keys()))
  data = [keys[i] for i in index]
  df = pd.DataFrame(data, columns=['Description'], index=index)
  return df

def render_html(test_sets):
  html = "<!DOCTYPE html><html><head></head><body>"
  info = info_table()
  html += info.to_html(escape=False)

  i = 0
  for test_set_name, df in test_sets.items():
    html += f"<hr/><br/><p><b>{test_set_name}</b></p><br/>"
    html += f'<div style="background-color:{get_color_name(i)};">'
    html += df.to_html(escape=False)
    html += '</div>'
    i += 1
  html += '</body></html>'

  output_file = 'index.html'
  with open(output_file, 'w') as f:
    f.write(html)

  webbrowser.open('file://' + os.path.realpath(output_file))