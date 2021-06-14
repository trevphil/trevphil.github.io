import os
import sys
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from collections import defaultdict
import uuid
from tqdm import tqdm

from generate_html import render_html, image_path_to_link

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Palatino"],
})

#################################################################
#################################################################
#################################################################

def get_subdirectories(basepath):
  x = [d for d in basepath.iterdir() if d.is_dir()]
  return list(sorted(x))


def get_files(basepath):
  x = [f for f in basepath.iterdir() if f.is_file()]
  return list(sorted(x))


def parse_summary_file(filepath):
  stats = defaultdict(lambda: np.nan)
  with open(filepath, 'r') as f:
    for line in f:
      name, val = line.split(':')[:2]
      stats[name.strip()] = float(val.strip())
  return stats

#################################################################
#################################################################
#################################################################

if os.path.exists('./sfm_results'):
  response = input('Delete existing results? [y/n] ')
  if response == 'y':
    if not os.path.exists('/tmp/sfm_results'):
      print('"/tmp/sfm_results" not found!')
      exit()
    shutil.rmtree('./sfm_results')
    shutil.copytree('/tmp/sfm_results', './sfm_results')
else:
  if not os.path.exists('/tmp/sfm_results'):
    print('"/tmp/sfm_results" not found!')
    exit()
  shutil.copytree('/tmp/sfm_results', './sfm_results')

for f in os.listdir('./plots'):
  if f.endswith('.png'):
    os.remove(os.path.join('./plots', f))

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

#################################################################
#################################################################
#################################################################

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

tmp_stats = parse_summary_file(test_dirs[0] / 'stats'/ 'summary.txt')
stat_names = set(tmp_stats.keys())
stat_names.add('Debug stereo')
stat_names.add('Debug depth')
stat_names = list(sorted(stat_names))

#################################################################
#################################################################
#################################################################

def make_dataframe():
  cols = stereo_methods
  rows = pd.MultiIndex.from_product([motions, stat_names])
  df = pd.DataFrame('-', rows, cols)
  return df

#################################################################
#################################################################
#################################################################

def parse_timeseries_file(filename):
  df = pd.read_csv(filename)
  df[df['Status'] != 'success'] = np.nan
  return df

def generate_timeseries_plot(data, column, ymin=None, ymax=None, title=None):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  for method, df in data.items():
    x = df['Frame ID']
    y = df[column]
    ax.plot(x, y, label=method, linewidth=0.7)
  ax.legend()
  ax.set_xlabel('Image sequence number')
  ax.set_ylabel(column)
  if (ymin is not None) and (ymax is not None):
    ax.set_ylim(ymin, ymax)
  if title is not None:
    ax.set_title(title)
  filename = './plots/%s.pdf' % str(uuid.uuid1())
  plt.savefig(filename)
  plt.close(fig=fig)
  return Path(filename)

#################################################################
#################################################################
#################################################################

test_sets = defaultdict(lambda: make_dataframe())  # Test set name --> table of results (dataframe)
timeseries = defaultdict(lambda: dict())  # (test set, motion) --> dict from (method name --> timeseries)
timeseries_cols = None
blacklisted_ts_cols = ('Frame ID', 'Failure', 'Status')

for test_dir in tqdm(test_dirs):
  debug_image_files = get_files(test_dir / 'debug')
  depth_image_files = get_files(test_dir / 'depth')

  if len(get_files(test_dir / 'stats')) == 0:
    print(f'No results for "{test_dir}"')
    sys.exit(1)

  stats = parse_summary_file(test_dir / 'stats' / 'summary.txt')

  motion = test_dir.parts[-1]
  test_set_name = test_dir.parts[-2]
  stereo_method = test_dir.parts[-3]

  df = test_sets[test_set_name]

  if stats['Failure'] == 1:
    df[stereo_method][motion]['Failure'] = 1
  else:
    for stat_name, stat_value in stats.items():
      df[stereo_method][motion][stat_name] = stat_value

  if len(debug_image_files) == 1:
    df[stereo_method][motion]['Debug stereo'] = image_path_to_link(debug_image_files[0])
  if len(depth_image_files) == 1:
    df[stereo_method][motion]['Debug depth'] = image_path_to_link(depth_image_files[0])

  ts = parse_timeseries_file(test_dir / 'stats' / 'timeseries.csv')
  if len(ts) > 1:
    timeseries[(test_set_name, motion)][stereo_method] = ts
    ts_cols = set([c for c in ts.columns if c not in blacklisted_ts_cols])
    if timeseries_cols is None:
      timeseries_cols = ts_cols
    else:
      timeseries_cols = timeseries_cols.intersection(ts_cols)

# Generate timeseries plots and add to summary tables
print('Generating plots...')
timeseries_cols = list(sorted(timeseries_cols))
for (test_set_name, motion), method2timeseries in timeseries.items():
  title = '%s: %s' % (test_set_name, motion)
  df = test_sets[test_set_name]
  if 'Timeseries' not in df.columns:
    df['Timeseries'] = '-'
  for col in timeseries_cols:
    image_path = generate_timeseries_plot(method2timeseries, col, title=title)
    df['Timeseries'][motion][col] = image_path_to_link(image_path)
  test_sets[test_set_name] = df

# Remove (outer) rows in each dataframe which have no data
print('Removing empty rows...')
all_motions = set([t.parts[-1] for t in test_dirs])
for test_set_name, df in test_sets.items():
  for m in all_motions:
    no_data = (df.loc[m].to_numpy().flatten() == '-').all()
    if no_data:
      df = df.drop(m, level=0)
  test_sets[test_set_name] = df


render_html(test_sets)
print('Done.')
sys.exit(0)
