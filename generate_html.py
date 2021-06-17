import os
import sys
import numpy as np
import pandas as pd
import webbrowser

N_NOISE_TRIALS = 50
INLIER_ERROR = 0.01
MIN_ESTIMATES_FOR_FAILURE = 0.01

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
    'Failure': f'1 if stereo failed OR if when projecting all estimated 3D points into the nearest (integer) pixel coordinate in the depth image, less than {MIN_ESTIMATES_FOR_FAILURE * 100.0}% of the image is "covered".',
    'Debug depth': 'A link to a "debug image" comparing ground truth vs. estimated depth. Coloring is based on z-coordinate in camera frame, NOT ray length from camera to surface, and is scaled by the overall (min, max) z-coordinate from estimated and ground truth points.',
    'Debug stereo': 'A link to a "debug image" from the stereo algorithm, this is specific and different for each algorithm.',
    'Fraction estimated px': 'Calculated by projecting all estimated 3D points into the nearest (integer) pixel coordinate in the depth image, and calculating how many pixels are filled',
    'Fraction inliers': f'The fraction of estimated points with relative error below {INLIER_ERROR * 100.0}%. Based on estimated vs. true ray length from camera origin to surface.',
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

def info_section():
  return f"""<h4>
  Values reported in the tables show μ ± 1-σ for each metric, when averaged over all
  frames (if the test is composed of more than 1 stereo image pair) and all trials.

  If a test had artificial noise added (pose or image noise), {N_NOISE_TRIALS} Monte Carlo trials were run.
  Otherwise for tests without noise, only 1 trial was run.

  The <em>Timeseries</em> plots show each metric over time, with dotted lines indicating ± 3-σ.
  The <em>Boxplot</em> plots show quartiles Q0, Q1, Q2, Q3, and Q4 for each metric, from all frames and trials.
  </h4>"""

def render_html(test_sets):
  html = "<!DOCTYPE html><html><head></head><body>"
  html += "<h4>The following metrics are computed on a <em>per-frame</em> basis (i.e. each time stereo runs on an image pair):</h4>"
  info = info_table()
  html += info.to_html(escape=False)
  html += "<hr/><br/>"
  html += info_section()

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