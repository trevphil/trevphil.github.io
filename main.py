import os
import sys
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from collections import defaultdict

from generate_html import render_html, image_path_to_link


def get_subdirectories(basepath):
  x = [d for d in basepath.iterdir() if d.is_dir()]
  return list(sorted(x))


def get_files(basepath):
  x = [f for f in basepath.iterdir() if f.is_file()]
  return list(sorted(x))


def parse_stats(filepath):
  key2name = {
    'complete_failure': 'Complete failure',
    'frac_estimated': 'Fraction estimated px',
    'frac_inliers': 'Fraction of inlier px',
    'frac_outliers': 'Fraction of outlier px',
    'outliers_over_inliers': 'Outliers / inliers',
    'max_error': 'Max abs. error [m]',
    'median_error': 'Median abs. error [m]',
    'rmse': 'RMSE [m]',
    'mean_relative_error': 'G.T. relative error'
  }
  stats = defaultdict(lambda: np.nan)
  with open(filepath, 'r') as f:
    for line in f:
      name, val, *_ = line.split(':')
      name = key2name.get(name, name)
      stats[name] = float(val)
  return stats


if not os.path.exists('/tmp/sfm_results'):
  print('"/tmp/sfm_results" not found!')
  exit()

if os.path.exists('./sfm_results'):
  response = input('Delete existing results? [y/n] ')
  if response != 'y':
    exit()
  shutil.rmtree('./sfm_results')

shutil.copytree('/tmp/sfm_results', './sfm_results')

basepath = Path('./sfm_results')
stereo_method_dirs = get_subdirectories(basepath)
if len(stereo_method_dirs) == 0:
  print('No stereo methods found!')
  sys.exit(1)

test_set_dirs = []
for stereo_method_dir in stereo_method_dirs:
  test_set_dirs += get_subdirectories(stereo_method_dir)
if len(test_set_dirs) == 0:
  print('No test sets found!')
  sys.exit(1)

test_dirs = []
for test_set_dir in test_set_dirs:
  test_dirs += get_subdirectories(test_set_dir)
if len(test_dirs) == 0:
  print('No tests found!')
  sys.exit(1)

print(f'Found {len(test_dirs)} total test(s)!')


def make_dataframe():
  stereo_methods = set()
  test_set_names = set()
  motions = set()

  for test_dir in test_dirs:
    motion = test_dir.parts[-1]
    test_set_name = test_dir.parts[-2]
    stereo_method = test_dir.parts[-3]

    motions.add(motion)
    test_set_names.add(test_set_name)
    stereo_methods.add(stereo_method)
  
  stereo_methods = list(sorted(stereo_methods))
  test_set_names = list(sorted(test_set_names))
  motions = list(sorted(motions))
  
  tmp_stats_files = get_files(test_dirs[0] / 'stats')
  tmp_stats = parse_stats(tmp_stats_files[0])
  stat_names = set(tmp_stats.keys())
  stat_names.add('Debug stereo')
  stat_names.add('Debug depth')
  stat_names = list(sorted(stat_names))

  cols = stereo_methods
  rows = pd.MultiIndex.from_product([motions, stat_names])
  df = pd.DataFrame('-', rows, cols)
  return df


test_sets = defaultdict(lambda: make_dataframe())

for test_dir in test_dirs:
  stats_files = get_files(test_dir / 'stats')
  debug_image_files = get_files(test_dir / 'debug')
  depth_image_files = get_files(test_dir / 'depth')

  if len(stats_files) == 0:
    print(f'No results for "{test_dir}"')
    sys.exit(1)
  elif len(stats_files) > 1:
    print('For now, the script only supports 1 frame per test!')
    sys.exit(1)

  stats = parse_stats(stats_files[0])
  stat_names = list(sorted(stats.keys()))

  motion = test_dir.parts[-1]
  test_set_name = test_dir.parts[-2]
  stereo_method = test_dir.parts[-3]
  
  df = test_sets[test_set_name]

  if stats['Complete failure'] == 1:
    df[stereo_method][motion]['Complete failure'] = 1
  else:
    for stat_name in stat_names:
      stat_value = stats[stat_name]
      df[stereo_method][motion][stat_name] = stat_value
  
  if len(debug_image_files) > 0:
    df[stereo_method][motion]['Debug stereo'] = image_path_to_link(debug_image_files[0])
  if len(depth_image_files) > 0:
    df[stereo_method][motion]['Debug depth'] = image_path_to_link(depth_image_files[0])

render_html(test_sets)

print('Done.')
sys.exit(0)
