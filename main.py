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
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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
  stats = defaultdict(lambda: defaultdict(lambda: np.nan))
  if not filepath.exists():
    return stats

  with open(filepath, 'r') as f:
    for line in f:
      parts = line.split(',')
      if len(parts) == 9:
        metric_name = parts[0].strip()
        n, mean, std, q00, q25, q50, q75, q100 = [float(p.strip()) for p in parts[1:]]
        stats[metric_name] = {
          'n': int(n),
          'mean': mean,
          'std': std,
          'q00': q00,
          'q25': q25,
          'q50': q50,
          'q75': q75,
          'q100': q100
        }
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
  if f.endswith('.pdf') or f.endswith('.png') or f.endswith('.jpg'):
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
  if 'JPLV' in stereo_method:
    stereo_method = 'JPLV Stereo (Normal Rect.)'
  elif 'General' in stereo_method:
    stereo_method = 'JPLV Stereo (General Rect.)'
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
  if not filename.exists():
    return None
  return pd.read_csv(filename, skipinitialspace=True)

def latex_sanitize(s):
  return s.replace('%', '\%').replace('<=', '$\le$').replace('>=', '$\ge$').replace('<', '$<$').replace('>', '$>$')

colors = [plt.get_cmap('Set2')(1. * i / len(stereo_methods)) for i in range(len(stereo_methods))]
method2color = dict(zip(stereo_methods, colors))

def generate_timeseries_plot(data, column, title=None):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  for method, (mean, std) in data.items():
    x = mean['Frame ID']
    y_center = mean[column]
    c = method2color[method]
    ax.plot(x, y_center, label=method, linewidth=1.0, c=c)
    if False: # np.sum(std[column]) > 1e-5:
      y_upper = y_center + (3 * std[column])
      y_lower = y_center - (3 * std[column])
      ax.plot(x, y_lower, linewidth=0.6, linestyle='dashed', c=c)
      ax.plot(x, y_upper, linewidth=0.6, linestyle='dashed', c=c)
  ax.legend()
  ax.set_xlabel('Image sequence number')
  ax.set_ylabel(latex_sanitize(column))
  if title is not None:
    ax.set_title(title)
  filename = './plots/%s.pdf' % str(uuid.uuid1())
  plt.savefig(filename)
  plt.close(fig=fig)
  return Path(filename)

def generate_boxplot(data, metric, title=None):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  x, labels, clrs = [], [], []
  for method, statistics in data.items():
    quartiles = [
      statistics[metric]['q00'],
      statistics[metric]['q25'],
      statistics[metric]['q50'],
      statistics[metric]['q75'],
      statistics[metric]['q100']
    ]
    x.append(quartiles)
    label = f'{method}\n[N={statistics[metric]["n"]}]'
    labels.append(label.replace(' (', '\n('))
    clrs.append(method2color[method])
  bplot = ax.boxplot(x, labels=labels, whis=[0, 100], patch_artist=True)
  for patch, c in zip(bplot['boxes'], clrs):
    patch.set_facecolor(c)
  plt.setp(bplot['medians'], color='black')
  ax.yaxis.grid(True)
  ax.set_ylabel(latex_sanitize(metric))
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
all_statistics = defaultdict(lambda: dict())  # (test_set, motion) --> dict from (method name --> statistics)
timeseries_cols = None
blacklisted_cols = ('Frame ID', 'Failure', 'Status')

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
  if 'JPLV' in stereo_method:
    stereo_method = 'JPLV Stereo (Normal Rect.)'
  elif 'General' in stereo_method:
    stereo_method = 'JPLV Stereo (General Rect.)'

  if test_set_name == 'AirSim Rocks':
    test_set_name = 'AirSim Rocks [No Noise]'

  df = test_sets[test_set_name]

  for stat_name, statistics in stats.items():
    df[stereo_method][motion][stat_name] = f"{statistics['mean']} ± {statistics['std']}"

  if len(debug_image_files) == 1:
    df[stereo_method][motion]['Debug stereo'] = image_path_to_link(debug_image_files[0])
  if len(depth_image_files) == 1:
    df[stereo_method][motion]['Debug depth'] = image_path_to_link(depth_image_files[0])

  ts_mean = parse_timeseries_file(test_dir / 'stats' / 'timeseries_mean.csv')
  ts_std = parse_timeseries_file(test_dir / 'stats' / 'timeseries_std.csv')
  if (ts_mean is not None) and (ts_std is not None) and \
      (len(ts_mean) > 1) and (len(ts_mean) == len(ts_std)):
    timeseries[(test_set_name, motion)][stereo_method] = (ts_mean, ts_std)
    ts_cols = set([c for c in ts_mean.columns if c not in blacklisted_cols])
    if timeseries_cols is None:
      timeseries_cols = ts_cols
    else:
      timeseries_cols = timeseries_cols.intersection(ts_cols)
  
  all_statistics[(test_set_name, motion)][stereo_method] = stats

# Generate timeseries plots and add to summary tables
print('Generating timeseries plots...')
timeseries_cols = list([s.strip() for s in sorted(timeseries_cols)])
for (test_set_name, motion), method2timeseries in timeseries.items():
  title = '%s: %s' % (test_set_name, motion)
  df = test_sets[test_set_name]
  if 'Timeseries' not in df.columns:
    df['Timeseries'] = '-'
  for col in timeseries_cols:
    image_path = generate_timeseries_plot(method2timeseries, col, title=title)
    df['Timeseries'][motion][col] = image_path_to_link(image_path)
  test_sets[test_set_name] = df

# Generate box-whisker plots and add to summary tables
print('Generating boxplots...')
for (test_set_name, motion), stats_per_method in all_statistics.items():
  title = '%s: %s' % (test_set_name, motion)
  df = test_sets[test_set_name]
  if 'Boxplot' not in df.columns:
    df['Boxplot'] = '-'
  for col in timeseries_cols:
    image_path = generate_boxplot(stats_per_method, col, title=title)
    df['Boxplot'][motion][col] = image_path_to_link(image_path)
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
