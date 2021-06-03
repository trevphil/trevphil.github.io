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


def parse_stats_file(filepath):
  stats = defaultdict(lambda: np.nan)
  with open(filepath, 'r') as f:
    for line in f:
      name, val, *_ = line.split(':')
      stats[name] = float(val)
  return stats


def parse_stats(stats_files):
  if len(stats_files) == 1:
    combined_stats = parse_stats_file(stats_files[0])
  else:
    all_stats = [parse_stats_file(f) for f in stats_files]
    succeeded = [s for s in all_stats if s['complete_failure'] == 0]
    n = float(len(all_stats))
    n_succeeded = float(len(succeeded))
    n_failed = n - n_succeeded
    combined_stats = defaultdict(lambda: np.nan)
    combined_stats['complete_failure'] = n_failed / n

    meank = lambda k: np.mean([s[k] for s in succeeded])
    mediank = lambda k: np.median([s[k] for s in succeeded])
    maxk = lambda k: np.max([s[k] for s in succeeded])

    if n_succeeded > 0:
      combined_stats['frac_estimated'] = meank('frac_estimated')
      combined_stats['frac_inliers'] = meank('frac_inliers')
      combined_stats['frac_outliers'] = 1.0 - combined_stats['frac_inliers']
      combined_stats['outliers_over_inliers'] = \
        combined_stats['frac_outliers'] / combined_stats['frac_inliers']
      combined_stats['max_error'] = maxk('max_error')
      combined_stats['median_error'] = mediank('median_error')
      combined_stats['rmse'] = meank('rmse')
      combined_stats['mean_relative_error'] = meank('mean_relative_error')

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
  final_output = defaultdict(lambda: np.nan)
  for k, val in combined_stats.items():
    name = key2name.get(k, k)
    final_output[name] = val
  return final_output


if not os.path.exists('/tmp/sfm_results'):
  print('"/tmp/sfm_results" not found!')
  exit()

if os.path.exists('./sfm_results'):
  response = input('Delete existing results? [y/n] ')
  if response == 'y':
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
  tmp_stats = parse_stats(tmp_stats_files)
  stat_names = set(tmp_stats.keys())
  stat_names.add('Debug stereo')
  stat_names.add('Debug depth')
  stat_names = list(sorted(stat_names))

  cols = stereo_methods
  rows = pd.MultiIndex.from_product([motions, stat_names])
  df = pd.DataFrame('-', rows, cols)
  return df


def get_stats_through_time(stats_files):
  did_fail = dict()
  rmse = dict()
  for stats_file in stats_files:
    idx = int(stats_file.parts[-1].split('.')[0])
    stats = parse_stats_file(stats_file)
    failed = int(stats['complete_failure'])
    did_fail[idx] = failed
    if failed == 0:
      rmse[idx] = stats['rmse']
  return did_fail, rmse


test_sets = defaultdict(lambda: make_dataframe())
failure_per_method = dict()
rmse_per_method = dict()

for test_dir in test_dirs:
  stats_files = get_files(test_dir / 'stats')
  debug_image_files = get_files(test_dir / 'debug')
  depth_image_files = get_files(test_dir / 'depth')

  if len(stats_files) == 0:
    print(f'No results for "{test_dir}"')
    sys.exit(1)

  stats = parse_stats(stats_files)
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

  if len(debug_image_files) == 1:
    df[stereo_method][motion]['Debug stereo'] = image_path_to_link(debug_image_files[0])
  if len(depth_image_files) == 1:
    df[stereo_method][motion]['Debug depth'] = image_path_to_link(depth_image_files[0])

  if len(stats_files) > 1:
    failures, rmse_errors = get_stats_through_time(stats_files)
    failure_per_method[stereo_method] = failures
    rmse_per_method[stereo_method] = rmse_errors


# Remove (outer) rows in each dataframe which have no data
all_motions = set([t.parts[-1] for t in test_dirs])
for test_set_name, df in test_sets.items():
  for m in all_motions:
    no_data = (df.loc[m].to_numpy().flatten() == '-').all()
    if no_data:
      df = df.drop(m, level=0)
  test_sets[test_set_name] = df


if len(failure_per_method) > 0:
  indices = set()
  for method, did_fail in failure_per_method.items():
    indices = set(did_fail.keys()).union(indices)
  tmax = int(np.max(list(indices))) + 1

  fig, axs = plt.subplots(len(failure_per_method), 1,
                          figsize=(10, 7))
  i = 0

  for method, did_fail in failure_per_method.items():
    successes = np.zeros(tmax, dtype=np.uint8)
    for idx, failed in did_fail.items():
      successes[int(idx)] = 1 - int(failed)
    axs[i].plot(successes, label=method)
    axs[i].legend()
    axs[i].set_xlabel('Image sequence number')
    axs[i].set_ylabel('Success?')
    axs[i].set_yticks([0, 1])
    i += 1

  plt.tight_layout()
  plt.savefig('success.jpg')

  max_rmse = 1.0

  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  for method, rmse in rmse_per_method.items():
    rmse_errors = np.zeros(tmax, dtype=float) * np.nan
    for idx, e in rmse.items():
      rmse_errors[int(idx)] = min(e, max_rmse)
    ax.plot(rmse_errors, label=method)
  ax.legend()
  ax.set_xlabel('Image sequence number')
  ax.set_ylabel('RMSE [m]')
  plt.savefig('rmse.jpg')


render_html(test_sets)
print('Done.')
sys.exit(0)
